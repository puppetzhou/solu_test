# PLM_Sol 使用指南

## 1. 工具简介

`PLM_Sol` 是一个基于 `ProtT5` 蛋白语言模型嵌入的蛋白可溶性预测工具。  
仓库路径：`/Volumes/T7_PuppetZ/solu_test/PLM_Sol`

该仓库的预测流程不是“直接输入 FASTA 后立刻输出结果”，而是分为两步：

1. 准备蛋白序列 `FASTA`
2. 用 `bio_embeddings` 先把 FASTA 转成 `ProtT5 embedding`
3. 再用 `PLM_Sol` 对生成的 embedding 做预测

## 2. 环境要求

根据仓库中的 `env.yml`、`requirements.txt` 和 `README.md`，核心环境如下：

- Python `3.8`
- PyTorch `2.0.1`
- torchvision `0.15.2`
- numpy `1.21.1`
- pandas `1.5.3`
- scipy `1.9.3`
- matplotlib `3.7.1`
- biopython `1.81`
- h5py `3.8.0`
- PyYAML `6.0`
- `bio-embeddings[all]`

README 中还注明了作者使用的是：

- `PyTorch 2.0.1`
- `CUDA 11.4`

## 3. Conda 环境创建与部署

建议在仓库根目录执行：

```bash
cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol

conda env create -f env.yml
conda activate PLM_Sol

pip install -r requirements.txt
pip install "bio-embeddings[all]"
```

如果你希望尽量贴近 README 中的 GPU 版本，可额外安装与本机 CUDA 匹配的 PyTorch。  
如果只是 CPU 运行，上述命令通常也可以完成部署，但 embedding 生成会慢很多。

## 4. 是否需要 GPU 加速

结论：

- 不是必须
- 但强烈建议在生成 `ProtT5 embedding` 时使用 GPU

原因：

- 代码中设备选择为自动检测：`torch.cuda.is_available()` 为真时使用 `cuda:0`，否则回退到 CPU
- 模型推理本身可在 CPU 上运行
- `ProtT5` embedding 生成最耗时，CPU 可跑但通常会明显更慢

因此：

- 小规模预测：可 CPU 运行
- 批量预测或长序列较多：建议使用 NVIDIA GPU

## 5. 预测输入文件格式

### 5.1 原始输入

原始输入是 `FASTA` 文件。

预测时，`PLM_Sol` 最终实际读取的不是原始 FASTA，而是：

- 一个 embedding 文件：`.h5`
- 一个 remapping 后的 FASTA：`.fasta`

这两个文件由 `bio_embeddings` 从原始 FASTA 自动生成。

### 5.2 FASTA 内容要求

序列部分应为标准蛋白氨基酸序列，例如：

```fasta
>protein_1
MNNIRRVAILLPFASASA
>protein_2
MSDHRAITLEGTTVMVPVYNSTAQ
```

建议注意：

- 每条序列的 FASTA header 第一列必须唯一
- header 尽量不要过于复杂
- 预测场景下不需要在 FASTA 头里附带标签

仓库中的训练集示例头部是：

```fasta
>SeCD00422904_pET21 A-0
MSDHRAITLEGTTVMVPVYNSTAQQSPYQLT...
```

其中 `A-0` 是训练标签信息；预测时不是必需的。

### 5.3 实际被模型读取的文件

预测脚本 `inference.py` 会读取：

- `embeddings`: `.h5`
- `remapping`: `.fasta`

默认配置文件在：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/configs/inference_Sol_biLSTM_TextCNN.yml`

## 6. 预测前需要改哪些路径

至少要改两个地方。

### 6.1 修改 embedding 生成配置

文件：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/embedding_dataset/embedding_protT5.yml`

需要修改：

- `global.sequences_file`
- `global.prefix`

示例：

```yml
global:
  sequences_file: ./my_predict.fasta
  prefix: ./my_predict_emb
```

含义：

- `sequences_file`：你的原始 FASTA
- `prefix`：embedding 输出目录前缀

运行后通常会得到类似：

- `./my_predict_emb/t5_embeddings/embeddings_file.h5`
- `./my_predict_emb/remapped_sequences_file.fasta`

### 6.2 修改推理配置

文件：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/configs/inference_Sol_biLSTM_TextCNN.yml`

至少需要检查或修改：

- `checkpoints_list`
- `embeddings`
- `remapping`
- `key_format`

推荐写法：

```yml
output_files_name: 'inference_on_custom'

log_iterations: 100
n_draws: 1000
batch_size: 1
checkpoints_list:
  - ./model_param/model_param.t7

embeddings: './embedding_dataset/my_predict_emb/t5_embeddings/embeddings_file.h5'
remapping: './embedding_dataset/my_predict_emb/remapped_sequences_file.fasta'
key_format: fasta_descriptor
```

说明：

- `checkpoints_list` 指向仓库自带模型权重
- `embeddings` 和 `remapping` 改成你自己的生成结果路径
- `key_format` 建议保持为 `fasta_descriptor`，与当前推理配置一致

## 7. 如何进行预测

### 7.1 准备输入 FASTA

将待预测序列保存为例如：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/embedding_dataset/my_predict.fasta`

### 7.2 生成 ProtT5 embedding

```bash
cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol/embedding_dataset
bio_embeddings embedding_protT5.yml
```

前提：你已经先把 `embedding_protT5.yml` 中的 `sequences_file` 和 `prefix` 改好。

### 7.3 执行预测

```bash
cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol
python inference.py --config ./configs/inference_Sol_biLSTM_TextCNN.yml
```

前提：你已经把 `configs/inference_Sol_biLSTM_TextCNN.yml` 中的：

- `embeddings`
- `remapping`
- `checkpoints_list`

改成可用路径。

### 7.4 预测输出

预测结果会写到仓库根目录下的：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/protTrans_prediction_result.csv`

输出列为：

- `protein_ID`
- `sequence`
- `predict_result`

其中 `predict_result` 是模型输出的预测分值。

## 8. 一套可直接参考的完整命令

假设你新建的输入文件为：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/embedding_dataset/my_predict.fasta`

则一个完整流程如下：

```bash
cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol

conda env create -f env.yml
conda activate PLM_Sol
pip install -r requirements.txt
pip install "bio-embeddings[all]"

cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol/embedding_dataset
bio_embeddings embedding_protT5.yml

cd /Volumes/T7_PuppetZ/solu_test/PLM_Sol
python inference.py --config ./configs/inference_Sol_biLSTM_TextCNN.yml
```

## 9. 可能需要微调或注意的地方

### 9.1 路径是最常见问题

本仓库默认配置里很多路径是示例路径，使用前要按你的文件实际位置替换。

重点检查：

- `embedding_dataset/embedding_protT5.yml`
- `configs/inference_Sol_biLSTM_TextCNN.yml`

### 9.2 权重文件位置

默认推理权重是：

- `/Volumes/T7_PuppetZ/solu_test/PLM_Sol/model_param/model_param.t7`

如果你换了模型权重，需要同步修改：

- `configs/inference_Sol_biLSTM_TextCNN.yml` 中的 `checkpoints_list`

### 9.3 批大小

如果显存不足，可以把：

- `configs/inference_Sol_biLSTM_TextCNN.yml` 中的 `batch_size`

设得更小，例如 `1`。

### 9.4 CPU 运行较慢

如果没有 GPU，最慢的环节通常是 `ProtT5 embedding` 生成，而不是最终分类预测。

## 10. 简短结论

对你的 4 个问题，可直接概括为：

1. 环境要求：`Python 3.8 + PyTorch 2.0.1 + bio-embeddings[all]`，建议用 `conda env create -f env.yml`
2. 是否需要 GPU：不是必须，但生成 `ProtT5 embedding` 时强烈建议 GPU
3. 预测输入格式：原始输入是 `FASTA`，模型实际读取的是 `bio_embeddings` 生成的 `.h5 + remapped fasta`
4. 如何预测：先改 `embedding_protT5.yml` 生成 embedding，再改 `inference_Sol_biLSTM_TextCNN.yml` 指向生成结果，最后运行 `python inference.py --config ...`
