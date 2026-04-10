"""
ESM特征提取脚本
从sufsol_all.csv中的蛋白质序列提取ESM2嵌入特征
"""

import torch
import esm
import pandas as pd
import numpy as np
import pickle
import os
from tqdm import tqdm


def extract_esm_features():
    """提取ESM特征并保存"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载ESM模型
    print("Loading ESM2 model...")
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()
    
    # 读取数据
    print("Loading protein data...")
    data = pd.read_csv('../S.cerevisiae_test.csv')
    print(f"Total proteins: {len(data)}")
    
    # 清理数据
    data = data.dropna(subset=['gene', 'sequence'])
    data['sequence'] = data['sequence'].astype(str).str.strip()
    data = data[data['sequence'].str.len() > 0]
    print(f"Valid proteins after cleaning: {len(data)}")
    
    # 限制序列长度（ESM模型有最大长度限制）
    MAX_LENGTH = 1024
    data['truncated_sequence'] = data['sequence'].str[:MAX_LENGTH]
    data['sequence_length'] = data['truncated_sequence'].str.len()
    
    valid_data = data[data['sequence_length'] > 0]
    sequences = [(row['gene'], row['truncated_sequence']) for _, row in valid_data.iterrows()]
    print(f"Sequences to process: {len(sequences)}")
    
    # 提取特征
    batch_size = 8  # 可以根据GPU内存调整
    sequence_representations = {}
    failed_sequences = []
    
    print("Extracting ESM features...")
    for i in tqdm(range(0, len(sequences), batch_size), desc="Processing Batches"):
        batch = sequences[i:i + batch_size]
        
        try:
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)
            batch_tokens = batch_tokens.to(device)
            
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[30])
            token_representations = results["representations"][30]
            
            for j, (gene_name, seq) in enumerate(batch):
                # 取序列的平均池化表示（去除CLS和SEP token）
                sequence_rep = token_representations[j, 1: len(seq) + 1].mean(0)
                sequence_representations[gene_name] = sequence_rep.cpu().numpy()
                
        except Exception as e:
            print(f"Failed to process batch starting with {batch[0][0]}: {e}")
            for gene_name, _ in batch:
                failed_sequences.append(gene_name)
    
    # 保存特征
    output_dir = 'esm_features'
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为pickle文件
    features_path = os.path.join(output_dir, 'esm_features_test.pkl')
    with open(features_path, 'wb') as f:
        pickle.dump(sequence_representations, f)
    
    # 保存为CSV（可选，用于查看）
    if sequence_representations:
        features_df = pd.DataFrame.from_dict(sequence_representations, orient='index')
        features_df.index.name = 'gene'
        features_df.to_csv(os.path.join(output_dir, 'esm_features_test.csv'))
    
    # 保存失败的序列列表
    if failed_sequences:
        failed_df = pd.DataFrame(failed_sequences, columns=['gene'])
        failed_df.to_csv(os.path.join(output_dir, 'failed_sequences.csv'), index=False)
    
    print(f"\nFeature extraction completed!")
    print(f"Successfully processed: {len(sequence_representations)} proteins")
    print(f"Failed sequences: {len(failed_sequences)}")
    print(f"ESM feature dimension: {list(sequence_representations.values())[0].shape[0] if sequence_representations else 'N/A'}")
    print(f"Features saved to: {features_path}")
    
    return sequence_representations


if __name__ == "__main__":
    extract_esm_features()
