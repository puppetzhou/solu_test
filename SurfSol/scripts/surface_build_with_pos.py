import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_cluster import radius
import glob
import os
import pandas as pd
# import pickle
import _pickle as pickle # use cPickle to speed up
from plyfile import PlyData
from torch_geometric.data import Data
from torch_geometric.transforms import FaceToEdge, Cartesian

def read_surface(surface_path,name):
    data_surf ={}
    print(len(glob.glob(f'{surface_path}/{name}')),surface_path,name)
    #if len(glob.glob(f'{surface_path}/{name}_*.ply'))==1:
    if len(glob.glob(f'{surface_path}/{name}'))==1:
        #for i in range(1):
            i=0
            #with open(glob.glob(f'{surface_path}/{name}_*.ply')[i], 'rb') as f:
            with open(glob.glob(f'{surface_path}/{name}')[i], 'rb') as f:    
                data = PlyData.read(f)
            features = ([torch.tensor(data['vertex'][axis.name]) for axis in data['vertex'].properties if axis.name not in ['nx', 'ny', 'nz'] ])
            pos = torch.stack(features[:3], dim=-1)
            # pos 需要减去center_protein_pos
            #pos -= complex_graph.original_center
            features = torch.stack(features[3:], dim=-1)
            face = None
            if 'face' in data:
                faces = data['face']['vertex_indices']
                faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
                face = torch.stack(faces, dim=-1)
            data = Data(x=features, pos=pos, face=face)
            data = FaceToEdge()(data)
            data = Cartesian(cat=False)(data)   
            data_surf['protein_'+str(i)] = data         
            data_surf['protein_'+str(i)].pos = data.pos
            data_surf['protein_'+str(i)].x = data.x
            data_surf['protein_'+str(i)].edge_index = data.edge_index
            data_surf['protein_'+str(i)].edge_attr = data.edge_attr
            return data_surf

def build_surface_conv_graph(data,id):
    # builds the receptor initial node and edge embeddings
    # tr = data['receptor'].node_t['tr']
    #tr = data[id].node_t['tr'][0]
    # data
    # data['surface'].node_t['tr'] = tr * torch.ones(data['surface'].num_nodes).to(tr.device)
    #data[id].node_sigma_emb = self.timestep_emb_func(tr * torch.ones(data['surface'].num_nodes).to(tr.device)) # tr rot and tor noise is all the same
    # surface may have nan in features
    node_attr = torch.nan_to_num(data[id].x)
    # this assumes the edges were already created in preprocessing since protein's structure is fixed
    edge_index = data[id].edge_index
    edge_attr = data[id].edge_attr.float()
    pos = data[id].pos  # 新增：獲取座標
    return node_attr, edge_index, edge_attr, pos




path = '/home/weizg/wei/soft/masif-master/data/masif_eSOL_ALA/data_preparation/01-benchmark_surfaces/'
filenames = os.listdir(path)
#listf=pd.read_csv('list')
#num=len(listf['pdb'])
for pdb in filenames:
#for i in range(0,100):
#    pdb=listf['pdb'][i]
#    print(pdb)
#    try:
        #info = pickle.load(open('data/graph_construct/intee_surf/' + pdb, 'rb'))
        data= read_surface('/home/weizg/wei/soft/masif-master/data/masif_eSOL_ALA/data_preparation/01-benchmark_surfaces/',pdb)
        print(data)
        # 修复：build_surface_conv_graph 返回顺序是: node_attr, edge_index, edge_attr, pos
        intra_surf1_x, intra_surf1_index, intra_surf1_attr, intra_surf1_pos = build_surface_conv_graph(data,'protein_0')
        intra_surf1_data = Data(x=intra_surf1_x, pos=intra_surf1_pos, edge_index=intra_surf1_index, edge_attr=intra_surf1_attr)
        
        #intra_surf1_x,intra_surf1_index,intra_surf1_attr= build_surface_conv_graph(data,'protein_0')
        
        #intra_surf2_x,intra_surf2_index,intra_surf2_attr= build_surface_conv_graph(data,'protein_1')
        #inter_surf_x,inter_surf_index,inter_surf_attr= build_surface_cross_conv_graph(data, cross_distance_cutoff=3.5)
    
        #intra_surf1_data = Data(x=intra_surf1_x, edge_index=intra_surf1_index, edge_attr=intra_surf1_attr)
        save_path = './surfgraph_with_pos/intra_surf1/' + pdb
        with open(save_path, 'wb') as f_save:
            pickle.dump(intra_surf1_data, f_save)
    
        #intra_surf2_data = Data(x=intra_surf2_x, edge_index=intra_surf2_index, edge_attr=intra_surf2_attr)
        #save_path = 'data/surfgraph/intra_surf2/' + pdb
        #with open(save_path, 'wb') as f_save:
        #    pickle.dump(intra_surf2_data, f_save)
    #
        #inter_surf_data = Data(x=inter_surf_x, edge_index=inter_surf_index, edge_attr=inter_surf_attr)
        #print(inter_surf_data)
        #save_path = 'data/surfgraph/inter_surf/' + pdb
        #with open(save_path, 'wb') as f_save:
        #    pickle.dump(inter_surf_data, f_save)  
#    except:
#        print(pdb,"ERROR") 

