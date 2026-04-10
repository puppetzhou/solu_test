import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius, radius_graph
from torch_scatter import scatter, scatter_mean
import numpy as np
from e3nn import o3
from e3nn.nn import BatchNorm
from e3nn.o3 import spherical_harmonics
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing, TransformerConv
from torch.nn import Embedding, Linear, ModuleList, Sequential
import pickle
import os


class MultiModalAttentionFusion(nn.Module):
    """Multi-modal attention fusion module"""
    def __init__(self, feature_dims, output_dim, num_heads=4, dropout=0.1):
        super(MultiModalAttentionFusion, self).__init__()
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Project each modality to the same dimension
        self.projectors = nn.ModuleList([
            nn.Linear(dim, output_dim) for dim in feature_dims
        ])
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, features):
        """
        Args:
            features: List of feature tensors from different modalities
        Returns:
            fused_features: Fused feature tensor
        """
        # Project all features to same dimension
        projected_features = []
        for i, feat in enumerate(features):
            proj_feat = self.projectors[i](feat)  # [batch, output_dim]
            projected_features.append(proj_feat)
        
        # Stack features for attention: [batch, num_modalities, output_dim]
        stacked_features = torch.stack(projected_features, dim=1)
        
        # Self-attention across modalities
        attended_features, attention_weights = self.attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Average pool across modalities
        fused_features = attended_features.mean(dim=1)  # [batch, output_dim]
        
        # Layer norm and dropout
        fused_features = self.layer_norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features


class AtomEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims, sigma_embed_dim=0, lm_embedding_type=None):
        super(AtomEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1] + sigma_embed_dim
        self.lm_embedding_type = lm_embedding_type
        
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)
            
        if self.lm_embedding_type is not None:
            if self.lm_embedding_type == 'esm':
                self.lm_embedding_dim = 1280
            else:
                raise ValueError('LM Embedding type was not correctly determined. LM embedding type: ', self.lm_embedding_type)
            self.lm_embedding_layer = torch.nn.Linear(self.lm_embedding_dim + emb_dim, emb_dim)
            
    def forward(self, x):
        x_embedding = 0
        if self.lm_embedding_type is not None:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features + self.lm_embedding_dim
        else:
            assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
            
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())

        if self.num_scalar_features > 0:
            x_embedding += self.linear(x[:, self.num_categorical_features:self.num_categorical_features + self.num_scalar_features])
            
        if self.lm_embedding_type is not None:
            x_embedding = self.lm_embedding_layer(torch.cat([x_embedding, x[:, -self.lm_embedding_dim:]], axis=1))
            
        return x_embedding


class TensorProductConvLayer(torch.nn.Module):
    def __init__(self, in_irreps, sh_irreps, out_irreps, n_edge_features, residual=True, batch_norm=True, dropout=0.0,
                 hidden_features=None):
        super(TensorProductConvLayer, self).__init__()
        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual
        if hidden_features is None:
            hidden_features = n_edge_features

        self.tp = tp = o3.FullyConnectedTensorProduct(in_irreps, sh_irreps, out_irreps, shared_weights=False)

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, hidden_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, tp.weight_numel)
        )
        self.batch_norm = BatchNorm(out_irreps) if batch_norm else None

    def forward(self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce='mean'):
        edge_src, edge_dst = edge_index
        tp = self.tp(node_attr[edge_dst], edge_sh, self.fc(edge_attr))

        out_nodes = out_nodes or node_attr.shape[0]
        out = scatter(tp, edge_src, dim=0, dim_size=out_nodes, reduce=reduce)

        if self.residual:
            padded = F.pad(node_attr, (0, out.shape[-1] - node_attr.shape[-1]))
            out = out + padded

        if self.batch_norm:
            out = self.batch_norm(out)
        return out


class SurfaceProteinProcessor:
    @staticmethod
    def read_surface(surface_path, name):
        data_surf = {}
        try:
            # 你的PLY文件格式是：基因名(小写)_A.ply
            surface_file = f'{surface_path}/{str(name).lower()}_A.ply'
            
            if not os.path.exists(surface_file):
                print(f"No surface file found for {name} at {surface_file}")
                return None
                
            # 直接用pickle加载，因为你的PLY文件就是pickle格式的Data对象
            with open(surface_file, 'rb') as f:
                data = pickle.load(f)
            
            # 根据你的surface_build.py，数据结构应该是：
            # Data(x=features, edge_index=edge_index, edge_attr=edge_attr)
            # 直接返回
            data_surf['surface'] = data
            return data_surf
            
        except Exception as e:
            print(f"Error reading surface for {name}: {e}")
            return None

    @staticmethod  
    def build_surface_conv_graph(data, id='surface'):
        """构建表面卷积图 - 完全按照你的surface_build.py"""
        # surface may have nan in features
        node_attr = torch.nan_to_num(data[id].x)
        # this assumes the edges were already created in preprocessing since protein's structure is fixed
        edge_index = data[id].edge_index
        edge_attr = data[id].edge_attr.float()
        
        # 提取节点位置信息（如果存在）
        pos = None
        if hasattr(data[id], 'pos') and data[id].pos is not None:
            pos = data[id].pos.float()
        
        return node_attr, edge_index, edge_attr, pos


class SurfSolProteinModel(torch.nn.Module):
    def __init__(self, config):
        super(SurfSolProteinModel, self).__init__()
        self.config = config
        if config.use_surface:
            self.surface_encoder = AtomEncoder(
                emb_dim=config.surface_emb_dim,
                feature_dims=([], 4),  
                lm_embedding_type=None
            )
            
            self.surface_conv_layers = nn.ModuleList()
            for i in range(config.num_conv_layers):
                self.surface_conv_layers.append(
                    TensorProductConvLayer(
                        in_irreps=f"{config.surface_emb_dim}x0e",
                        sh_irreps="1x0e+1x1e+1x2e",
                        out_irreps=f"{config.surface_emb_dim}x0e",
                        n_edge_features=config.surface_edge_dim,
                        batch_norm=True,
                        dropout=config.dropout
                    )
                )
        
        # ESM特征编码器（如果使用）
        if config.use_esm:
            self.esm_encoder = nn.Linear(config.esm_dim, config.esm_emb_dim)
        
        # 3D结构编码器
        if config.use_structure:
            self.structure_net = StructureNet(config)
        
        # 融合层 - 支持多模态
        fusion_input_dim = 0
        feature_dims = []
        
        if config.use_surface:
            fusion_input_dim += config.surface_emb_dim
            feature_dims.append(config.surface_emb_dim)
        if config.use_esm:
            fusion_input_dim += config.esm_emb_dim
            feature_dims.append(config.esm_emb_dim)
        if config.use_structure:
            fusion_input_dim += config.structure_emb_dim
            feature_dims.append(config.structure_emb_dim)
            
        assert fusion_input_dim > 0, "At least one modality must be enabled!"
        
        # Choose fusion method based on config
        if config.fusion_type == "attention" and len(feature_dims) > 1:
            # Use attention fusion for multi-modal case
            self.fusion_module = MultiModalAttentionFusion(
                feature_dims=feature_dims,
                output_dim=config.hidden_dim,
                num_heads=config.attention_heads,
                dropout=config.dropout
            )
            fusion_output_dim = config.hidden_dim
        else:
            # Use concatenation fusion (default or single modal)
            self.fusion_module = None
            fusion_output_dim = fusion_input_dim
        
        # Final MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_output_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # 输出头
        if config.training_mode in ["classification", "regression+classification"] and config.use_classification_head:
            # 包含分类任务：需要分类头
            self.classification_head = nn.Linear(config.hidden_dim, 1)  # 二分类（sigmoid输出）
        else:
            self.classification_head = None
            
        if config.training_mode in ["regression", "regression+classification"]:
            # 包含回归任务：需要回归头
            self.regression_head = nn.Linear(config.hidden_dim, 1)
        else:
            self.regression_head = None
        
    def forward(self, hetero_data):
        """前向传播 - 支持多模态"""
        feature_list = []
        
        # 1. 处理表面特征
        if self.config.use_surface and 'surface' in hetero_data.node_types:
            surface_x = self.surface_encoder(hetero_data['surface'].x)
            surface_edge_index = hetero_data['surface', 'surface_edge', 'surface'].edge_index
            surface_edge_attr = hetero_data['surface', 'surface_edge', 'surface'].edge_attr
            
            # 获取节点位置信息（如果存在）
            surface_pos = None
            if hasattr(hetero_data['surface'], 'pos') and hetero_data['surface'].pos is not None:
                surface_pos = hetero_data['surface'].pos
            
            # 表面卷积
            for conv_layer in self.surface_conv_layers:
                # 优先使用真实的3D位置计算球面谐波
                edge_sh = self._compute_spherical_harmonics(
                    surface_edge_index, 
                    surface_pos, 
                    sh_irreps="1x0e+1x1e+1x2e"
                )
                
                # 如果没有pos，回退到简单方法
                if edge_sh is None:
                    edge_sh = self._create_simple_edge_features(surface_edge_attr)
                
                surface_x = conv_layer(surface_x, surface_edge_index, surface_edge_attr, edge_sh)
            
            # 表面全局池化
            surface_batch = hetero_data['surface'].batch
            surface_repr = scatter_mean(surface_x, surface_batch, dim=0)  # [batch_size, surface_emb_dim]
            feature_list.append(surface_repr)
        
        # 2. ESM特征处理
        if self.config.use_esm and 'esm' in hetero_data.node_types and hetero_data['esm'].x.numel() > 0:
            esm_features = hetero_data['esm'].x  # [batch_size, esm_dim]
            esm_repr = self.esm_encoder(esm_features)  # [batch_size, esm_emb_dim]
            feature_list.append(esm_repr)
        
        # 3. 3D结构特征处理 (新增)
        if self.config.use_structure and 'structure' in hetero_data.node_types:
            structure_data = hetero_data['structure']
            structure_repr = self.structure_net(
                pos=structure_data.pos,
                a=structure_data.a,
                cc=structure_data.cc,
                dh=structure_data.dh,
                batch=structure_data.batch
            )  # [batch_size, structure_emb_dim]
            feature_list.append(structure_repr)
        
        # 4. Multi-modal fusion
        assert len(feature_list) > 0, "No valid modality features!"
        
        if self.fusion_module is not None and len(feature_list) > 1:
            # Use attention fusion
            fused_features = self.fusion_module(feature_list)
        else:
            # Use concatenation fusion
            combined_repr = torch.cat(feature_list, dim=-1)
            fused_features = combined_repr
        
        # Apply final MLP
        fused_features = self.fusion_mlp(fused_features)
        
        # 5. 最终预测
        outputs = {}
        
        # 回归输出（如果启用）
        if self.regression_head is not None:
            regression_output = self.regression_head(fused_features)
            outputs['regression'] = regression_output
        
        # 分类输出（如果启用）
        if self.classification_head is not None:
            classification_logits = self.classification_head(fused_features)
            outputs['classification'] = classification_logits
        
        return outputs
    
    def _compute_spherical_harmonics(self, edge_index, pos, sh_irreps="1x0e+1x1e+1x2e"):
        """
        使用真实的3D位置信息计算球面谐波特征
        
        Args:
            edge_index: [2, E] 边索引
            pos: [N, 3] 节点3D坐标
            sh_irreps: 球面谐波的不可约表示字符串，例如 "1x0e+1x1e+1x2e" 表示0阶+1阶+2阶
        
        Returns:
            edge_sh: [E, sh_dim] 每条边的球面谐波特征
        """
        if pos is None:
            # 如果没有pos，回退到使用edge_attr的简单方法
            return None
        
        edge_src, edge_dst = edge_index
        
        # 计算从源节点到目标节点的方向向量
        vec = pos[edge_dst] - pos[edge_src]  # [E, 3]
        
        # 计算距离
        distances = torch.norm(vec, dim=-1, keepdim=True)  # [E, 1]
        
        # 避免除零错误：对于距离为0的边，使用单位向量
        mask = distances.squeeze() < 1e-6
        if mask.any():
            # 对于距离为0的边，使用随机方向或默认方向
            vec[mask] = torch.tensor([1.0, 0.0, 0.0], device=vec.device)
            distances[mask] = 1.0
        
        # 归一化方向向量（单位向量）
        vec_normalized = vec / distances  # [E, 3]
        
        # 使用 e3nn 计算球面谐波
        # sh_irreps="1x0e+1x1e+1x2e" 表示：
        # - 1x0e: 1个0阶（l=0）的偶函数（scalar）
        # - 1x1e: 1个1阶（l=1）的偶函数（vector，3维）
        # - 1x2e: 1个2阶（l=2）的偶函数（tensor，5维）
        # 总共：1 + 3 + 5 = 9维
        irreps = o3.Irreps(sh_irreps)
        edge_sh = spherical_harmonics(irreps, vec_normalized, normalize=True, normalization='component')
        
        return edge_sh
    
    def _create_simple_edge_features(self, edge_attr):
        """
        创建简单的边特征（回退方法，当没有pos时使用）
        使用边属性创建简单的特征来近似球面谐波
        """
        # 使用边属性创建简单的特征
        num_edges = edge_attr.shape[0]
        device = edge_attr.device  # 获取edge_attr的设备
        
        # 创建简单的边特征 (1 + 3 + 5 = 9维，对应0阶+1阶+2阶球面谐波的维度)
        sh_0 = torch.ones(num_edges, 1, device=device)  # 0阶，指定设备
        
        # 使用边属性作为1阶特征
        if edge_attr.shape[1] >= 3:
            sh_1 = edge_attr[:, :3]  # 1阶，使用前3个边属性
        else:
            sh_1 = torch.zeros(num_edges, 3, device=device)  # 指定设备
        
        # 创建简单的2阶特征
        sh_2 = torch.zeros(num_edges, 5, device=device)  # 指定设备
        if edge_attr.shape[1] >= 3:
            # 使用边属性的组合作为2阶特征
            sh_2[:, 0] = edge_attr[:, 0] * edge_attr[:, 1] if edge_attr.shape[1] > 1 else 0
            sh_2[:, 1] = edge_attr[:, 1] * edge_attr[:, 2] if edge_attr.shape[1] > 2 else 0
            sh_2[:, 2] = edge_attr[:, 0] * edge_attr[:, 2] if edge_attr.shape[1] > 2 else 0
            sh_2[:, 3] = edge_attr[:, 0]**2 - edge_attr[:, 1]**2 if edge_attr.shape[1] > 1 else 0
            sh_2[:, 4] = edge_attr[:, 2]**2 if edge_attr.shape[1] > 2 else 0
        
        return torch.cat([sh_0, sh_1, sh_2], dim=-1)


class ShiftedSoftplus(torch.nn.Module):
    """Shifted Softplus activation function."""
    def __init__(self):
        super().__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class GaussianSmearing(torch.nn.Module):
    """Gaussian smearing of interatomic distances."""
    def __init__(self, start=0.0, stop=5.0, num_gaussians=50):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        dist = dist.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(dist, 2))


def radius_graph_custom(x, r, batch=None, loop=False, max_num_neighbors=32, flow='source_to_target'):
    """Custom radius_graph implementation to avoid pytorch bugs."""
    if x.numel() == 0:
        return torch.empty(2, 0, dtype=torch.long, device=x.device)
    
    from torch_cluster import radius_graph
    try:
        return radius_graph(x, r, batch, loop, max_num_neighbors, flow)
    except:
        # Fallback implementation
        num_nodes = x.size(0)
        edge_indices = []
        
        for i in range(num_nodes):
            distances = torch.norm(x - x[i], dim=1)
            neighbors = torch.where(distances < r)[0]
            if not loop:
                neighbors = neighbors[neighbors != i]
            neighbors = neighbors[:max_num_neighbors]
            
            for j in neighbors:
                if flow == 'source_to_target':
                    edge_indices.append([i, j])
                else:
                    edge_indices.append([j, i])
        
        if edge_indices:
            return torch.tensor(edge_indices).t().contiguous()
        else:
            return torch.empty(2, 0, dtype=torch.long, device=x.device)


class ResidualBlock(nn.Module):
    """Residual GNN block."""
    def __init__(self, block, dim):
        super(ResidualBlock, self).__init__()
        self.block = block
        self.layernorm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def reset_parameters(self):
        self.block.reset_parameters()

    def forward(self, x, edge_index, edge_weight, edge_attr):
        h = self.block(x, edge_index, edge_weight, edge_attr)
        h = self.layernorm(h)
        h = self.act(h)
        return h


class CustomInteractionBlock(nn.Module):
    """Custom interaction block for a GNN using TransformerConv."""
    def __init__(self, hidden_channels, num_gaussians, num_filters, heads):
        super().__init__()
        self.mlp = Sequential(
            Linear(num_gaussians, num_filters),
            ShiftedSoftplus(),
            Linear(num_filters, num_filters),
        )
        self.conv = TransformerConv(hidden_channels, hidden_channels // heads,
                                    heads, edge_dim=num_filters)
        self.act = ShiftedSoftplus()
        self.lin = Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.mlp[0].weight)
        self.mlp[0].bias.data.fill_(0)
        torch.nn.init.xavier_uniform_(self.mlp[2].weight)
        self.mlp[2].bias.data.fill_(0)
        self.conv.reset_parameters()
        torch.nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_weight, edge_attr):
        e = self.mlp(edge_attr)
        x = self.conv(x, edge_index, e)
        x = self.act(x)
        x = self.lin(x)
        return x


class StructureNet(nn.Module):
    """3D结构编码网络 - 照搬自参考代码Net类"""
    def __init__(self, config):
        super().__init__()
        
        # 配置参数
        self.param = nn.Parameter(torch.empty(0))  # Dummy parameter
        self.hidden_channels = config.structure_hidden_channels
        self.num_filters = config.structure_num_filters
        self.num_interactions = config.structure_num_interactions
        self.num_gaussians = config.structure_num_gaussians
        self.cutoff = config.structure_cutoff
        self.max_num_neighbors = config.structure_max_num_neighbors
        
        # 激活函数
        self.act = ShiftedSoftplus()
        
        # 嵌入层
        self.embedding = Embedding(20, self.hidden_channels)  # 20种氨基酸
        
        # CA-CofM距离编码 (使用RBF)
        cc_gaussians = 500
        self.cc_rbf = GaussianSmearing(0.0, 150.0, cc_gaussians)
        self.embed_cc = nn.Sequential(
            nn.Linear(cc_gaussians, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # 二面角特征编码
        self.embed_dh = nn.Sequential(
            nn.Linear(15, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # 节点特征融合
        self.embed_node = nn.Sequential(
            nn.Linear(self.hidden_channels * 3, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.hidden_channels)
        )
        
        # 边特征编码
        self.distance_expansion = GaussianSmearing(0.0, self.cutoff, self.num_gaussians)
        
        # GNN交互层
        self.interactions = ModuleList()
        for _ in range(self.num_interactions):
            block = CustomInteractionBlock(
                self.hidden_channels, self.num_gaussians, 
                self.num_filters, heads=1
            )
            self.interactions.append(ResidualBlock(block, self.hidden_channels))
        
        # 最终投影层
        self.final_projection = nn.Linear(self.hidden_channels, config.structure_emb_dim)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """重置参数"""
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()
    
    def forward(self, pos, a, cc, dh, batch=None):
        """
        前向传播
        Args:
            pos: [N, 3] CA原子坐标
            a: [N] 氨基酸类型
            cc: [N, 1] CA到质心距离 
            dh: [N, 15] 二面角特征
            batch: [N] batch指示器
        """
        batch = torch.zeros_like(a) if batch is None else batch
        
        # 特征嵌入
        h = self.embedding(a)  # [N, hidden_channels]
        
        # CA-CofM距离标准化和编码
        mu = 28.612717
        sigma = 18.466433
        cc_norm = (cc - mu) / sigma
        j = self.cc_rbf(cc_norm)
        j = self.embed_cc(j)  # [N, hidden_channels]
        
        # 二面角编码
        k = self.embed_dh(dh)  # [N, hidden_channels]
        
        # 节点特征融合
        h = self.embed_node(torch.cat([h, j, k], dim=1))  # [N, hidden_channels]
        
        # 构建图边
        edge_index = radius_graph_custom(pos, r=self.cutoff, batch=batch,
                                       max_num_neighbors=self.max_num_neighbors)
        row, col = edge_index
        dists = (pos[row] - pos[col]).norm(dim=-1)
        edge_weight = dists
        edge_attr = self.distance_expansion(dists).to(self.param.device)
        
        # GNN交互层
        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)
        
        # 全局池化
        h_global = scatter(h, batch, dim=0, reduce='mean')  # [batch_size, hidden_channels]
        
        # 最终投影
        structure_repr = self.final_projection(h_global)  # [batch_size, structure_emb_dim]
        
        return structure_repr


class SurfSolConfig:
    """SurfSol模型配置 - 适配你的数据格式"""
    def __init__(self):
        # 模态开关
        self.use_surface = True
        self.use_esm = True
        self.use_structure = True
        
        # 表面网络参数
        self.surface_emb_dim = 32
        self.surface_edge_dim = 3
        self.num_conv_layers = 2
        
        # ESM参数
        self.esm_dim = 640
        self.esm_emb_dim = 32
        
        # 3D结构网络参数
        self.structure_hidden_channels = 150
        self.structure_num_filters = 150
        self.structure_num_interactions = 6
        self.structure_num_gaussians = 300
        self.structure_cutoff = 15.0
        self.structure_max_num_neighbors = 150
        self.structure_emb_dim = 32
        
        # 融合层参数
        self.hidden_dim = 256
        self.dropout = 0.3
        
        # 融合方式配置
        self.fusion_type = "concat"  # "concat" or "attention"
        self.attention_heads = 4  # 注意力头数（仅在attention模式下使用）
        
        # 训练模式配置
        self.training_mode = "regression"  # "regression", "classification", or "regression+classification"
        self.use_classification_head = True  # 是否使用分类头
        self.classification_weight = 1.0  # 分类损失权重 (α)
        self.regression_weight = 1.0  # 回归损失权重 (β)
        self.classification_threshold = 1.0  # 二分类阈值 (1.0表示溶解度==1.0 vs <1.0)
        
        # 重采样配置
        self.use_resampling = True  # 是否使用重采样
        self.oversample_range = (0.4, 0.8)  # 过采样范围
        self.oversample_factor = 3  # 过采样倍数
        self.undersample_target = 1.0  # 欠采样目标值
        self.undersample_factor = 0.5  # 欠采样比例
        
        # 分阶段学习配置
        self.use_staged_learning = False  # 是否使用分阶段学习
        self.stage1_epochs = 50  # 第一阶段（二分类）轮数
        self.stage2_epochs = 30  # 第二阶段（细分回归）轮数
        self.stage3_epochs = 20  # 第三阶段（联合预测）轮数
        
        # 训练参数
        self.learning_rate = 1e-3
        self.weight_decay = 1e-4
        self.patience = 20
        self.batch_size = 5
        
        # Pearson相关损失参数
        self.use_pearson_loss = True  # 是否使用Pearson相关损失
        self.pearson_weight = 0.1  # Pearson损失权重
