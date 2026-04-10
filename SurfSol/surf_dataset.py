import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Batch, HeteroData
import glob
import os
import pandas as pd
from surf_model import SurfaceProteinProcessor
import mdtraj as md
from mdtraj import load_pdb
from mdtraj import compute_center_of_mass as calc_cofm
from tqdm import tqdm
import hashlib
import json
from datetime import datetime


class ProteinStructureProcessor:
    """3D structure processor - based on reference code NumpyRep"""
    @staticmethod
    def process_pdb(pdb_path):
        """Process PDB file and extract 3D structure features"""
        try:
            traj = load_pdb(pdb_path)
            
            # Get CA atom coordinates
            ca_indices = [i for i, atom in enumerate(traj.topology.atoms) if atom.name == 'CA']
            pos = traj.xyz[0][ca_indices, :] * 10  # nm -> Å
            
            # Get amino acid types
            AAs = ['GLN','TRP','GLU','ARG','THR','TYR','ILE','PRO',
                   'ALA','SER','ASP','PHE','GLY','HIS','LYS','LEU',
                   'CYS','VAL','ASN','MET']
            aa_map = {aa: i for i, aa in enumerate(AAs)}
            aa_map['HIS'] = aa_map.get('HIS', 13)  # Default histidine mapping
            
            a = np.array([aa_map.get(r.name, aa_map['HIS']) for r in traj.topology.residues 
                         if list(r.atoms_by_name('CA'))])
            
            # Calculate CA to center of mass distance
            cofm = calc_cofm(traj)[0] * 10  # nm -> Å
            pos_minus_cofm = pos - cofm[np.newaxis, :]
            cc = np.sqrt(np.einsum('ij,ij->i', pos_minus_cofm, pos_minus_cofm))
            
            # Calculate dihedral angle features
            dh = ProteinStructureProcessor._get_dihedral_features(traj)
            
            return {
                'pos': torch.tensor(pos, dtype=torch.float32),
                'a': torch.tensor(a, dtype=torch.long), 
                'cc': torch.tensor(cc, dtype=torch.float32).unsqueeze(1),
                'dh': torch.tensor(dh, dtype=torch.float32)
            }
            
        except Exception as e:
            print(f"Error processing PDB {pdb_path}: {e}")
            return None
    
    @staticmethod
    def _get_dihedral_features(traj):
        """Calculate dihedral angle features - 15 dimensions"""
        # Create atom to residue mapping
        a2r = {}
        i = 0
        for r in traj.topology.residues:
            if not list(r.atoms_by_name('CA')):
                continue
            else:
                for a in r.atoms:
                    a2r[a.index] = i
                i += 1
        
        num_residues = i
        
        # Calculate Psi angles
        try:
            psis = np.array(list(md.compute_psi(traj)[1][0]) + [-2*np.pi])
        except:
            psis = np.zeros(num_residues) - 2*np.pi
        psis_mask = np.ones(len(psis))
        psis_mask[len(psis) - 1] = 0
        psis_sin = np.sin(psis)
        psis_sin[len(psis) - 1] = 0
        psis_cos = np.cos(psis)
        psis_cos[len(psis) - 1] = 0
        
        # Calculate Phi angles
        try:
            phis = np.array([-2*np.pi] + list(md.compute_phi(traj)[1][0]))
        except:
            phis = np.zeros(num_residues) - 2*np.pi
        phis_mask = np.ones(len(phis))
        phis_mask[0] = 0
        phis_sin = np.sin(phis)
        phis_sin[0] = 0
        phis_cos = np.cos(phis)
        phis_cos[0] = 0
        
        # Calculate Chi1 angles
        chi1s = np.zeros(num_residues) + 10
        chi1s_mask = np.ones(num_residues)
        try:
            chi1_data = md.compute_chi1(traj)
            for i, chi in enumerate(chi1_data[1][0]):
                chi1s[a2r[chi1_data[0][i][0]]] = chi
        except:
            pass
        
        chi1s_sin = np.sin(chi1s)
        chi1s_cos = np.cos(chi1s)
        for i, chi in enumerate(chi1s):
            if chi == 10:
                chi1s_mask[i] = 0
                chi1s_sin[i] = 0
                chi1s_cos[i] = 0
        
        # Calculate Chi2 angles  
        chi2s = np.zeros(num_residues) + 10
        chi2s_mask = np.ones(num_residues)
        try:
            chi2_data = md.compute_chi2(traj)
            for i, chi in enumerate(chi2_data[1][0]):
                chi2s[a2r[chi2_data[0][i][0]]] = chi
        except:
            pass
            
        chi2s_sin = np.sin(chi2s)
        chi2s_cos = np.cos(chi2s)
        for i, chi in enumerate(chi2s):
            if chi == 10:
                chi2s_mask[i] = 0
                chi2s_sin[i] = 0
                chi2s_cos[i] = 0
        
        # Calculate Chi3 angles
        chi3s = np.zeros(num_residues) + 10
        chi3s_mask = np.ones(num_residues)
        try:
            chi3_data = md.compute_chi3(traj)
            for i, chi in enumerate(chi3_data[1][0]):
                chi3s[a2r[chi3_data[0][i][0]]] = chi
        except:
            pass
            
        chi3s_sin = np.sin(chi3s)
        chi3s_cos = np.cos(chi3s)
        for i, chi in enumerate(chi3s):
            if chi == 10:
                chi3s_mask[i] = 0
                chi3s_sin[i] = 0
                chi3s_cos[i] = 0
        
        # Combine all features [15 dimensions]
        dh_features = np.array([
            psis_mask, psis_sin, psis_cos,
            phis_mask, phis_sin, phis_cos, 
            chi1s_mask, chi1s_sin, chi1s_cos,
            chi2s_mask, chi2s_sin, chi2s_cos,
            chi3s_mask, chi3s_sin, chi3s_cos
        ]).transpose()
        
        return dh_features


class SurfSolDataset(Dataset):
    """SurfSol-based dataset - supports tri-modal with resampling"""
    def __init__(self, csv_data, surface_path, esm_features=None, pdb_path=None, cache_dir="data_cache", config=None):
        """
        Initialize dataset
        Args:
            csv_data: sufsol_all.csv data
            surface_path: surface data path
            esm_features: ESM feature dictionary
            pdb_path: PDB file path (new)
            cache_dir: directory to store cached preprocessed data
            config: SurfSolConfig object for resampling parameters
        """
        self.csv_data = csv_data
        self.surface_path = surface_path
        self.esm_features = esm_features if esm_features is not None else {}
        self.pdb_path = pdb_path  # New: PDB path
        self.cache_dir = cache_dir
        self.config = config
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Filter valid data with progress bar
        self.valid_data = []
        print("Processing dataset...")
        for idx, row in tqdm(csv_data.iterrows(), total=len(csv_data), desc="Validating data files", ncols=100):
            protein_name = row['gene']
            if self._check_data_exists(protein_name):
                self.valid_data.append({
                    'protein': protein_name,
                    'solubility': row['solubility'],
                    'sequence': row['sequence'] if 'sequence' in row else None
                })
        
        print(f"Valid data: {len(self.valid_data)}/{len(csv_data)}")
        
        # Store original data for cache signature (before resampling)
        self.original_data = self.valid_data.copy()
        
        # Load or preprocess 3D structure data with caching BEFORE resampling
        self.structure_cache = {}
        if self.pdb_path:
            cache_loaded = self._load_structure_cache()
            if not cache_loaded:
                self._preprocess_structures()
                self._save_structure_cache()
        
        # Apply resampling if enabled (AFTER structure preprocessing)
        if self.config and self.config.use_resampling:
            self.valid_data = self._apply_resampling(self.valid_data)
            print(f"After resampling: {len(self.valid_data)} samples")

    def _check_data_exists(self, protein_name):
        """Check if necessary data files exist"""
        # Surface file must exist
        surface_file = f'{self.surface_path}/{str(protein_name).lower()}_A.ply'
        if not os.path.exists(surface_file):
            return False
        
        # PDB file check (if pdb_path provided)
        if self.pdb_path:
            pdb_file = f'{self.pdb_path}/{str(protein_name).lower()}.pdb'
            if not os.path.exists(pdb_file):
                print(f"Warning: PDB file not found for {protein_name}")
                # Option: return False to force PDB file existence
                # Here we choose to continue but not load 3D structure features
        
        return True
        
    def _apply_resampling(self, data_list):
        """Apply resampling strategy to balance the dataset"""
        if not self.config:
            return data_list
            
        print("🔄 Applying resampling strategy...")
        
        # 分析原始数据分布
        solubilities = [item['solubility'] for item in data_list]
        
        # 统计不同区间的数据
        in_range_count = sum(1 for s in solubilities if self.config.oversample_range[0] <= s <= self.config.oversample_range[1])
        target_count = sum(1 for s in solubilities if s == self.config.undersample_target)
        other_count = len(solubilities) - in_range_count - target_count
        
        print(f"   Original distribution:")
        print(f"   - Range {self.config.oversample_range}: {in_range_count} samples")
        print(f"   - Target value {self.config.undersample_target}: {target_count} samples")
        print(f"   - Others: {other_count} samples")
        
        resampled_data = []
        
        # 1. 过采样：对0.4-0.8范围的数据进行过采样
        oversample_data = [item for item in data_list 
                          if self.config.oversample_range[0] <= item['solubility'] <= self.config.oversample_range[1]]
        
        # 添加原始数据
        for item in oversample_data:
            resampled_data.append(item)
            # 添加过采样副本
            for i in range(self.config.oversample_factor - 1):
                resampled_data.append(item.copy())
        
        # 2. 欠采样：对溶解度=1的数据进行欠采样
        target_data = [item for item in data_list if item['solubility'] == self.config.undersample_target]
        undersample_count = int(len(target_data) * self.config.undersample_factor)
        
        # 随机选择欠采样数据
        import random
        random.seed(42)  # 保证可重现性
        undersampled_target = random.sample(target_data, min(undersample_count, len(target_data)))
        resampled_data.extend(undersampled_target)
        
        # 3. 保留其他数据不变
        other_data = [item for item in data_list 
                     if not (self.config.oversample_range[0] <= item['solubility'] <= self.config.oversample_range[1]) 
                     and item['solubility'] != self.config.undersample_target]
        resampled_data.extend(other_data)
        
        # 统计重采样后的分布
        new_solubilities = [item['solubility'] for item in resampled_data]
        new_in_range = sum(1 for s in new_solubilities if self.config.oversample_range[0] <= s <= self.config.oversample_range[1])
        new_target = sum(1 for s in new_solubilities if s == self.config.undersample_target)
        new_other = len(new_solubilities) - new_in_range - new_target
        
        print(f"   Resampled distribution:")
        print(f"   - Range {self.config.oversample_range}: {new_in_range} samples (↑{new_in_range/in_range_count:.1f}x)")
        print(f"   - Target value {self.config.undersample_target}: {new_target} samples (↓{new_target/target_count:.1f}x)")
        print(f"   - Others: {new_other} samples")
        
        return resampled_data
        
        
    def _preprocess_structures(self):
        """Preprocess all 3D structure data with detailed progress tracking"""
        print("🧬 Processing 3D protein structures...")
        
        # Get unique protein names from original data to avoid processing duplicates
        unique_proteins = list(set(item['protein'] for item in self.original_data))
        
        successful_count = 0
        failed_count = 0
        
        # Process each unique protein with progress bar
        pbar = tqdm(unique_proteins, desc="Processing PDB structures", ncols=120, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
        
        for protein_name in pbar:
            pdb_file = f'{self.pdb_path}/{str(protein_name).lower()}.pdb'
            
            if os.path.exists(pdb_file):
                structure_data = ProteinStructureProcessor.process_pdb(pdb_file)
                
                if structure_data is not None:
                    self.structure_cache[protein_name] = structure_data
                    successful_count += 1
                else:
                    self.structure_cache[protein_name] = None
                    failed_count += 1
            else:
                self.structure_cache[protein_name] = None
                failed_count += 1
                
            # Update progress bar with current stats
            pbar.set_postfix({
                'Success': successful_count,
                'Failed': failed_count,
                'Rate': f'{successful_count/(successful_count+failed_count)*100:.1f}%'
            })
        
        # Final statistics
        print(f"\n📊 3D Structure Processing Results:")
        print(f"   ✅ Successfully processed: {successful_count} proteins")
        print(f"   ❌ Failed/Missing: {failed_count} proteins") 
        print(f"   📈 Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    
    def _get_cache_signature(self):
        """Generate a signature for cache validation"""
        # Always use original data (before resampling) for cache signature
        unique_proteins = sorted(set(item['protein'] for item in self.original_data))
        signature_data = {
            'proteins': unique_proteins,
            'pdb_path': self.pdb_path,
            'total_proteins': len(unique_proteins)
        }
        signature_str = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_str.encode()).hexdigest()
    
    def _get_cache_paths(self):
        """Get cache file paths"""
        signature = self._get_cache_signature()
        cache_file = os.path.join(self.cache_dir, f"structure_cache_{signature}.pkl")
        meta_file = os.path.join(self.cache_dir, f"structure_meta_{signature}.json")
        return cache_file, meta_file
    
    def _load_structure_cache(self):
        """Load cached structure data if available and valid"""
        cache_file, meta_file = self._get_cache_paths()
        
        if not (os.path.exists(cache_file) and os.path.exists(meta_file)):
            print("🔍 No valid cache found, will preprocess structures...")
            return False
        
        try:
            # Load metadata
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            
            print(f"🔍 Found cache from {meta_data['created_time']}")
            print(f"   Cache contains {meta_data['total_proteins']} proteins")
            print(f"   Success rate: {meta_data['success_rate']:.1f}%")
            
            # Load cached structure data
            print("📁 Loading cached structure data...")
            with open(cache_file, 'rb') as f:
                self.structure_cache = pickle.load(f)
            
            print("✅ Successfully loaded cached structure data!")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cache: {e}")
            print("🔄 Will preprocess structures from scratch...")
            return False
    
    def _save_structure_cache(self):
        """Save processed structure data to cache"""
        cache_file, meta_file = self._get_cache_paths()
        
        try:
            print("💾 Saving structure cache...")
            
            # Save structure data
            with open(cache_file, 'wb') as f:
                pickle.dump(self.structure_cache, f)
            
            # Count successful structures
            successful_count = sum(1 for v in self.structure_cache.values() if v is not None)
            total_count = len(self.structure_cache)
            
            # Save metadata
            meta_data = {
                'created_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_proteins': total_count,
                'successful_count': successful_count,
                'success_rate': successful_count / total_count * 100 if total_count > 0 else 0,
                'pdb_path': self.pdb_path,
                'signature': self._get_cache_signature()
            }
            
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            print(f"✅ Cache saved successfully!")
            print(f"   Cache file: {cache_file}")
            print(f"   Metadata: {meta_file}")
            
        except Exception as e:
            print(f"❌ Failed to save cache: {e}")

    def __len__(self):
        return len(self.valid_data)

    def __getitem__(self, idx):
        item = self.valid_data[idx]
        protein_name = item['protein']
        
        # Create heterogeneous graph
        hetero_data = HeteroData()
        
        # 1. Load surface data
        surface_data = SurfaceProteinProcessor.read_surface(self.surface_path, protein_name)
        if surface_data is None:
            hetero_data['surface'].x = torch.zeros(1, 4)
            hetero_data['surface', 'surface_edge', 'surface'].edge_index = torch.zeros(2, 0, dtype=torch.long)
            hetero_data['surface', 'surface_edge', 'surface'].edge_attr = torch.zeros(0, 3)
        else:
            surf_data = surface_data['surface']
            hetero_data['surface'].x = surf_data.x
            
            # 检测 edge_index 和 pos 是否被交换
            edge_index_raw = surf_data.edge_index
            pos_raw = getattr(surf_data, 'pos', None)
            
            # 如果 edge_index 是 [E, 3] 且 pos 是 [2, N]，且 E == N，可能被交换了
            swapped = False
            if (edge_index_raw.dim() == 2 and edge_index_raw.shape[1] == 3 and 
                pos_raw is not None and pos_raw.dim() == 2 and pos_raw.shape[0] == 2 and
                edge_index_raw.shape[0] == pos_raw.shape[1]):
                # 检查是否应该交换：edge_index 的 [E, 3] 可能是 pos，pos 的 [2, N] 可能是 edge_index
                # 如果 pos 转置后可以 reshape 成 [N, 3]，则交换
                pos_t = pos_raw.t()
                if pos_t.numel() % 3 == 0:
                    print(f"Warning: Detected swapped edge_index and pos for {protein_name}, fixing...")
                    edge_index_raw, pos_raw = pos_raw, edge_index_raw[:, :3]  # 取前3列作为pos
                    swapped = True
            
            # 确保 edge_index 的形状是 [2, E]，如果不是则转置
            edge_index = edge_index_raw
            
            # 确保 edge_index 是 2D 张量
            if edge_index.dim() != 2:
                raise ValueError(f"Edge index must be 2D, got shape: {edge_index.shape} for protein {protein_name}")
            
            # 确保形状是 [2, E]
            if edge_index.shape[0] != 2:
                if edge_index.shape[1] == 2:
                    # 如果是 [E, 2]，转置为 [2, E]
                    edge_index = edge_index.t().contiguous()
                elif edge_index.shape[1] == 3 and edge_index.shape[0] > 2:
                    # 如果是 [E, 3]，可能前两列是源和目标节点，提取前两列并转置
                    print(f"Warning: Edge index has shape {edge_index.shape} for protein {protein_name}, extracting first 2 columns")
                    edge_index = edge_index[:, :2].t().contiguous()
                else:
                    # 如果形状不对，创建空的 edge_index
                    print(f"Warning: Edge index has wrong shape {edge_index.shape} for protein {protein_name}, creating empty edge_index")
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=edge_index.device)
            
            # 确保 edge_index 的值在有效范围内
            num_nodes = surf_data.x.shape[0]
            if edge_index.numel() > 0:
                edge_index = torch.clamp(edge_index, 0, num_nodes - 1)
            
            # 最终验证形状
            if edge_index.shape[0] != 2:
                raise ValueError(f"After processing, edge_index still has wrong shape: {edge_index.shape}, expected [2, E] for protein {protein_name}")
            
            hetero_data['surface', 'surface_edge', 'surface'].edge_index = edge_index
            hetero_data['surface', 'surface_edge', 'surface'].edge_attr = surf_data.edge_attr
            # 如果存在 pos，也保存它（虽然当前模型不使用，但保持数据完整性）
            if pos_raw is not None:
                pos = pos_raw
                # 确保 pos 是 2D 张量，形状为 [N, 3]
                if pos.dim() == 1:
                    # 如果是 1D，尝试 reshape 为 [N, 3]
                    if pos.numel() % 3 == 0:
                        pos = pos.view(-1, 3)
                    else:
                        print(f"Warning: pos for {protein_name} has shape {pos.shape}, cannot reshape to [N, 3], skipping pos")
                        pos = None
                elif pos.dim() == 2:
                    # 确保第二个维度是3
                    if pos.shape[1] != 3:
                        # 如果 pos 是 [2, N] 格式，可能是数据被错误存储，尝试转置
                        if pos.shape[0] == 2 and pos.shape[1] > 3:
                            print(f"Warning: pos for {protein_name} has shape {pos.shape}, transposing to [N, 2] but cannot reshape to [N, 3], skipping pos")
                            pos = None
                        elif pos.shape[0] == 3 and pos.shape[1] > 3:
                            # 如果是 [3, N]，转置为 [N, 3]
                            print(f"Warning: pos for {protein_name} has shape {pos.shape}, transposing to [N, 3]")
                            pos = pos.t().contiguous()
                        else:
                            print(f"Warning: pos for {protein_name} has shape {pos.shape}, expected [N, 3], reshaping if possible")
                            if pos.numel() % 3 == 0:
                                pos = pos.view(-1, 3)
                            else:
                                print(f"Cannot reshape, skipping pos for {protein_name}")
                                pos = None
                else:
                    print(f"Warning: pos for {protein_name} has unexpected dimension {pos.dim()}, skipping")
                    pos = None
                
                if pos is not None:
                    # 确保 pos 是 float 类型（torch.norm 需要浮点数）
                    if not pos.is_floating_point():
                        pos = pos.float()
                    hetero_data['surface'].pos = pos

        # 2. Load ESM features
        if protein_name in self.esm_features:
            esm_feature = torch.tensor(self.esm_features[protein_name], dtype=torch.float32)
            hetero_data['esm'].x = esm_feature.unsqueeze(0)
        else:
            hetero_data['esm'].x = torch.zeros(1, 1280)

        # 3. Load 3D structure features from cache
        if self.pdb_path and protein_name in self.structure_cache:
            structure_data = self.structure_cache[protein_name]
            
            if structure_data is not None:
                hetero_data['structure'].pos = structure_data['pos']
                hetero_data['structure'].a = structure_data['a']
                hetero_data['structure'].cc = structure_data['cc']
                hetero_data['structure'].dh = structure_data['dh']
            else:
                # Create empty 3D structure placeholder
                hetero_data['structure'].pos = torch.zeros(1, 3)
                hetero_data['structure'].a = torch.zeros(1, dtype=torch.long)
                hetero_data['structure'].cc = torch.zeros(1, 1)
                hetero_data['structure'].dh = torch.zeros(1, 15)
        else:
            # If no PDB path provided or not cached, create empty placeholders
            hetero_data['structure'].pos = torch.zeros(1, 3)
            hetero_data['structure'].a = torch.zeros(1, dtype=torch.long)
            hetero_data['structure'].cc = torch.zeros(1, 1)
            hetero_data['structure'].dh = torch.zeros(1, 15)

        # 4. Label processing
        solubility = item['solubility']
        regression_target = torch.tensor([solubility], dtype=torch.float32)
        
        # Use configurable threshold for classification
        threshold = self.config.classification_threshold if self.config else 1.0
        classification_target = torch.tensor([1.0 if solubility >= threshold else 0.0], dtype=torch.float32)

        return {
            'hetero_data': hetero_data,
            'regression_target': regression_target,
            'classification_target': classification_target,
            'protein_name': protein_name
        }


def surf_collate_fn(batch):
    """Collate function for heterogeneous graphs - supports tri-modal"""
    hetero_batch = []
    regression_targets = []
    classification_targets = []
    protein_names = []
    
    for item in batch:
        hetero_batch.append(item['hetero_data'])
        regression_targets.append(item['regression_target'])
        classification_targets.append(item['classification_target'])
        protein_names.append(item['protein_name'])
    
    # Batch heterogeneous graphs
    batched_hetero = HeteroData()
    
    # 1. Process surface data
    surface_x_list = [data['surface'].x for data in hetero_batch]
    surface_edge_index_list = []
    surface_edge_attr_list = []
    
    node_offset = 0
    for i, data in enumerate(hetero_batch):
        edge_index = data['surface', 'surface_edge', 'surface'].edge_index
        edge_attr = data['surface', 'surface_edge', 'surface'].edge_attr
        num_nodes = data['surface'].x.shape[0]
        
        # 获取设备（如果 edge_index 为空，使用 edge_attr 的设备，否则使用 edge_index 的设备）
        device = edge_index.device if edge_index.numel() > 0 else (edge_attr.device if edge_attr.numel() > 0 else torch.device('cpu'))
        
        # 确保 edge_index 的形状是 [2, E]
        if edge_index.numel() == 0:
            # 如果 edge_index 为空，创建空的 [2, 0] 形状
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr_dim = edge_attr.shape[1] if edge_attr.numel() > 0 else 3
            edge_attr = torch.zeros(0, edge_attr_dim, dtype=torch.float32, device=device)
        else:
            # 确保 edge_index 是 2D 张量
            if edge_index.dim() != 2:
                raise ValueError(f"Edge index must be 2D, got shape: {edge_index.shape} for sample {i}")
            
            # 检查并修正 edge_index 的形状：必须是 [2, E]
            if edge_index.shape[0] != 2:
                if edge_index.shape[1] == 2:
                    # 如果是 [E, 2]，转置为 [2, E]
                    edge_index = edge_index.t().contiguous()
                else:
                    # 如果形状不对，创建空的 edge_index
                    edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
                    edge_attr = torch.zeros(0, edge_attr.shape[1] if edge_attr.numel() > 0 else 3, dtype=torch.float32, device=device)
            
            # 确保 edge_index 的值在有效范围内
            if edge_index.shape[1] > 0:
                edge_index = torch.clamp(edge_index, 0, num_nodes - 1)
                
                # 确保 edge_attr 的形状与 edge_index 匹配
                num_edges = edge_index.shape[1]
                if edge_attr.numel() == 0:
                    # 如果 edge_attr 为空，创建默认值
                    edge_attr_dim = 3
                    edge_attr = torch.zeros(num_edges, edge_attr_dim, dtype=torch.float32, device=device)
                elif edge_attr.shape[0] != num_edges:
                    # 如果数量不匹配，调整 edge_attr
                    if edge_attr.shape[0] > num_edges:
                        edge_attr = edge_attr[:num_edges]
                    else:
                        # 填充 edge_attr
                        padding_size = num_edges - edge_attr.shape[0]
                        edge_attr_dim = edge_attr.shape[1]
                        padding = torch.zeros(padding_size, edge_attr_dim, dtype=edge_attr.dtype, device=device)
                        edge_attr = torch.cat([edge_attr, padding], dim=0)
        
        # 确保 edge_index 的形状正确（在添加偏移之前）
        if edge_index.dim() != 2:
            raise ValueError(f"Edge index must be 2D before offset, got shape: {edge_index.shape} for sample {i}")
        if edge_index.shape[0] != 2:
            raise ValueError(f"Edge index must have shape [2, E] before offset, got shape: {edge_index.shape} for sample {i}")
        
        # 应用节点偏移（即使是空的 edge_index 也要添加，以保持一致性）
        edge_index_offset = edge_index.clone()
        if edge_index.shape[1] > 0:
            # 确保 node_offset 是标量
            if not isinstance(node_offset, (int, torch.Tensor)):
                raise ValueError(f"node_offset must be int or Tensor, got {type(node_offset)}")
            if isinstance(node_offset, torch.Tensor):
                node_offset = node_offset.item()
            edge_index_offset = edge_index_offset + node_offset
        
        # 再次确保维度正确：必须是 [2, E]
        if edge_index_offset.dim() != 2:
            raise ValueError(f"After offset, edge_index has wrong dimension: {edge_index_offset.dim()}, expected 2D for sample {i}")
        if edge_index_offset.shape[0] != 2:
            raise ValueError(f"After offset, edge_index has wrong shape: {edge_index_offset.shape}, expected [2, E] for sample {i}")
        
        surface_edge_index_list.append(edge_index_offset)
        surface_edge_attr_list.append(edge_attr)
        
        node_offset += num_nodes
    
    batched_hetero['surface'].x = torch.cat(surface_x_list, dim=0)
    
    if len(surface_edge_index_list) > 0:
        # 确保所有 edge_index 的第一个维度都是 2，第二个维度可以不同
        for idx, ei in enumerate(surface_edge_index_list):
            if ei.dim() != 2:
                raise ValueError(f"Edge index {idx} must be 2D, got {ei.dim()}D with shape {ei.shape}")
            if ei.shape[0] != 2:
                # 如果形状不对，尝试转置
                if ei.shape[1] == 2:
                    surface_edge_index_list[idx] = ei.t().contiguous()
                    print(f"Warning: Edge index {idx} was [E, 2], transposed to [2, E]")
                else:
                    raise ValueError(f"Edge index {idx} has wrong shape: {ei.shape}, expected [2, E]. Cannot auto-fix.")
        
        # 再次检查所有 edge_index 的形状（在转置后）
        for idx, ei in enumerate(surface_edge_index_list):
            if ei.shape[0] != 2:
                raise ValueError(f"After processing, edge index {idx} still has wrong shape: {ei.shape}, expected [2, E]")
        
        # 确保所有 edge_index 都在同一个设备上
        target_device = surface_edge_index_list[0].device
        surface_edge_index_list = [ei.to(target_device) for ei in surface_edge_index_list]
        surface_edge_attr_list = [ea.to(target_device) for ea in surface_edge_attr_list]
        
        # 确保所有 edge_index 的 dtype 一致
        target_dtype = surface_edge_index_list[0].dtype
        surface_edge_index_list = [ei.to(target_dtype) for ei in surface_edge_index_list]
        
        # 现在可以安全地拼接
        try:
            batched_hetero['surface', 'surface_edge', 'surface'].edge_index = torch.cat(surface_edge_index_list, dim=1)
            batched_hetero['surface', 'surface_edge', 'surface'].edge_attr = torch.cat(surface_edge_attr_list, dim=0)
        except RuntimeError as e:
            # 提供更详细的错误信息
            shapes = [ei.shape for ei in surface_edge_index_list]
            raise RuntimeError(
                f"Failed to concatenate edge_index tensors. Shapes: {shapes}. "
                f"All tensors must have shape [2, E] where E can vary. Error: {e}"
            )
    else:
        batched_hetero['surface', 'surface_edge', 'surface'].edge_index = torch.zeros(2, 0, dtype=torch.long)
        batched_hetero['surface', 'surface_edge', 'surface'].edge_attr = torch.zeros(0, 3)
    
    # 处理 surface pos（如果存在）
    # 收集所有有 pos 的数据
    surface_pos_list = []
    for data in hetero_batch:
        if hasattr(data['surface'], 'pos') and data['surface'].pos is not None:
            surface_pos_list.append(data['surface'].pos)
    
    # 只有当所有样本都有 pos 时才拼接
    if len(surface_pos_list) == len(hetero_batch) and len(surface_pos_list) > 0:
        
        # 验证所有 pos 的形状
        pos_shapes = [pos.shape for pos in surface_pos_list]
        if len(pos_shapes) > 0:
            # 确保所有 pos 都是 2D 张量
            for idx, pos in enumerate(surface_pos_list):
                if pos.dim() != 2:
                    raise ValueError(
                        f"Surface pos {idx} must be 2D tensor, got shape: {pos.shape}. "
                        f"All pos shapes: {pos_shapes}"
                    )
            
            # 检查第二个维度是否一致（坐标维度，通常是3）
            if len(set(pos.shape[1] for pos in surface_pos_list if pos.dim() == 2)) > 1:
                shapes_str = ', '.join([str(s) for s in pos_shapes])
                raise ValueError(
                    f"Surface pos tensors have inconsistent second dimension. "
                    f"All pos must have same coordinate dimension (e.g., [N, 3]). "
                    f"Shapes: {shapes_str}"
                )
            
            # 确保所有 pos 都在同一个设备上
            target_device = surface_pos_list[0].device
            surface_pos_list = [pos.to(target_device) for pos in surface_pos_list]
            
            # 确保所有 pos 都是 float 类型（torch.norm 需要浮点数）
            surface_pos_list = [pos.float() if not pos.is_floating_point() else pos for pos in surface_pos_list]
            
            # 确保所有 pos 的 dtype 一致（现在应该都是 float32）
            target_dtype = surface_pos_list[0].dtype
            surface_pos_list = [pos.to(target_dtype) for pos in surface_pos_list]
        
        try:
            batched_hetero['surface'].pos = torch.cat(surface_pos_list, dim=0)
        except RuntimeError as e:
            shapes_str = ', '.join([str(pos.shape) for pos in surface_pos_list])
            raise RuntimeError(
                f"Failed to concatenate surface pos tensors. "
                f"Shapes: {shapes_str}. "
                f"All tensors must have same shape except dimension 0. Error: {e}"
            )
    
    # 2. Process ESM data
    esm_x_list = [data['esm'].x for data in hetero_batch]
    batched_hetero['esm'].x = torch.cat(esm_x_list, dim=0)
    
    # 3. Process 3D structure data (new)
    structure_pos_list = [data['structure'].pos for data in hetero_batch]
    structure_a_list = [data['structure'].a for data in hetero_batch]
    structure_cc_list = [data['structure'].cc for data in hetero_batch]
    structure_dh_list = [data['structure'].dh for data in hetero_batch]
    
    batched_hetero['structure'].pos = torch.cat(structure_pos_list, dim=0)
    batched_hetero['structure'].a = torch.cat(structure_a_list, dim=0)
    batched_hetero['structure'].cc = torch.cat(structure_cc_list, dim=0)
    batched_hetero['structure'].dh = torch.cat(structure_dh_list, dim=0)
    
    # 4. Create batch indicators
    surface_batch = []
    esm_batch = []
    structure_batch = []
    
    for i, data in enumerate(hetero_batch):
        surface_batch.extend([i] * data['surface'].x.shape[0])
        esm_batch.extend([i] * data['esm'].x.shape[0])
        structure_batch.extend([i] * data['structure'].pos.shape[0])
    
    batched_hetero['surface'].batch = torch.tensor(surface_batch, dtype=torch.long)
    batched_hetero['esm'].batch = torch.tensor(esm_batch, dtype=torch.long)
    batched_hetero['structure'].batch = torch.tensor(structure_batch, dtype=torch.long)
    
    return {
        'hetero_data': batched_hetero,
        'regression_targets': torch.stack(regression_targets),
        'classification_targets': torch.stack(classification_targets),
        'protein_names': protein_names
    }


def load_surface_esm_data(csv_path, surface_path, esm_features_path=None, pdb_path=None, cache_dir="data_cache", config=None):
    """Load complete dataset - supports tri-modal with caching and resampling"""
    import pandas as pd
    
    # Load CSV data
    csv_data = pd.read_csv(csv_path)
    print(f"Loading CSV data: {len(csv_data)} records")
    
    # Load ESM features
    esm_features = {}
    if esm_features_path and os.path.exists(esm_features_path):
        try:
            with open(esm_features_path, 'rb') as f:
                esm_features = pickle.load(f)
            print(f"Loading ESM features: {len(esm_features)} proteins")
        except Exception as e:
            print(f"Failed to load ESM features: {e}")
    
    # Create dataset (with cache support and resampling)
    dataset = SurfSolDataset(csv_data, surface_path, esm_features, pdb_path, cache_dir, config)
    
    return dataset


def load_train_test_data(train_csv_path, test_csv_path, surface_path, esm_features_path=None, pdb_path=None, cache_dir="data_cache", config=None):
    """Load separate training and test datasets"""
    import pandas as pd
    
    # Load training data
    train_data = pd.read_csv(train_csv_path)
    print(f"Loading training data: {len(train_data)} records from {train_csv_path}")
    
    # Load test data  
    test_data = pd.read_csv(test_csv_path)
    print(f"Loading test data: {len(test_data)} records from {test_csv_path}")
    
    # Load ESM features
    esm_features = {}
    if esm_features_path and os.path.exists(esm_features_path):
        try:
            with open(esm_features_path, 'rb') as f:
                esm_features = pickle.load(f)
            print(f"Loading ESM features: {len(esm_features)} proteins")
        except Exception as e:
            print(f"Failed to load ESM features: {e}")
    
    # Create separate datasets (disable resampling for test set)
    train_config = config
    test_config = config.__class__() if config else None
    if test_config and config:
        # Copy config but disable resampling for test set
        for attr_name in dir(config):
            if not attr_name.startswith('_'):
                setattr(test_config, attr_name, getattr(config, attr_name))
        test_config.use_resampling = False
    
    train_dataset = SurfSolDataset(train_data, surface_path, esm_features, pdb_path, cache_dir, train_config)
    test_dataset = SurfSolDataset(test_data, surface_path, esm_features, pdb_path, cache_dir, test_config)
    
    return train_dataset, test_dataset
