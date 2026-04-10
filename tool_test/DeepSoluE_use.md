# DeepSoluE 使用指南

## 1. 软件简介
DeepSoluE 是一个蛋白质可溶性预测工具。仓库中的本地可运行主程序位于：

`/Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code`

主入口为：

`DeepSoluE.py`

它会先从蛋白序列生成多组特征，再调用 10 个 TensorFlow 模型集成预测，最终输出 `.csv` 结果。

## 2. 环境要求

### 2.1 README 中明确给出的依赖
- Python `3.7.3` 或以上
- `gensim==3.4.0`
- `pandas==1.1.3`
- `tensorflow==2.3.0`
- `biopython==1.78`
- `numpy==1.19.2`

### 2.2 从代码补充识别出的依赖
仓库代码还额外用到了以下包：

- `scikit-learn`
- `joblib`

原因：
- `feature_scripts/feature_combine.py` 中调用了 `joblib.load('./model/scaler.gz')`
- 同文件还导入了 `sklearn`

### 2.3 外部软件要求
DeepSoluE 本地完整预测还依赖两个未随仓库提供的外部程序：

- `TMHMM`
- `USEARCH`

代码默认期望它们位于：

- `./softs/tmhmm-2.0c/bin/tmhmm`
- `./softs/usearch/usearch`

此外，USEARCH 还需要数据库 FASTA：

- `./softs/usearch/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta`

注意：仓库根目录下虽然已经有一份
`/Volumes/T7_PuppetZ/solu_test/DeepSoluE/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta`
但代码实际读取的是 `DeepSoluE-master_source_code/softs/usearch/` 下的同名文件，因此需要复制过去，或改代码路径。

## 3. 是否需要 GPU
不强制需要 GPU。

结论如下：
- 代码里没有任何“必须使用 GPU”的显式设置。
- `TMHMM` 和 `USEARCH` 都是 CPU 程序。
- TensorFlow 预测阶段可以在 CPU 上运行，只是速度可能比 GPU 慢。

因此：
- 小规模预测：CPU 就可以运行。
- 大批量预测：GPU 可选，但不是必需条件。

## 4. 推荐部署方式

### 4.1 建议工作目录
实际运行时必须进入：

```bash
cd /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code
```

原因：仓库大量使用相对路径，例如 `./sequence/`、`./features/`、`./model/`、`./softs/`。如果不在该目录执行，程序会找不到文件。

### 4.2 conda 环境创建
建议命令：

```bash
conda create -n deepsolue python=3.7.3 -y
conda activate deepsolue
pip install numpy==1.19.2 pandas==1.1.3 biopython==1.78 gensim==3.8.3 scikit-learn==0.23.2 joblib==0.17.0 tensorflow==2.3.0
```

说明：
- README 写的是 `gensim==3.4.0`，但同类老项目常见地用 `3.x` 均可；若你希望完全贴 README，可改成 `gensim==3.4.0`。
- 若 `tensorflow==2.3.0` 在当前系统安装困难，优先考虑 Linux/x86_64 环境。

### 4.3 部署外部程序
在 `DeepSoluE-master_source_code` 下准备如下目录结构：

```text
DeepSoluE-master_source_code/
├── softs/
│   ├── tmhmm-2.0c/
│   │   └── bin/
│   │       └── tmhmm
│   └── usearch/
│       ├── usearch
│       └── Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta
```

可执行部署动作示例：

```bash
cd /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code
mkdir -p softs/usearch softs/tmhmm-2.0c/bin
cp /你的下载路径/usearch softs/usearch/
cp /你的下载路径/tmhmm softs/tmhmm-2.0c/bin/
cp /Volumes/T7_PuppetZ/solu_test/DeepSoluE/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta softs/usearch/
chmod +x softs/usearch/usearch
chmod +x softs/tmhmm-2.0c/bin/tmhmm
```

## 5. 输入文件格式
预测输入是蛋白质序列 `FASTA` 文件，不是 CSV、TSV 或其他结构化表格。

要求：
- 文件格式：`.fasta`
- 文件内容：标准蛋白序列
- 存放位置：必须放到 `DeepSoluE-master_source_code/sequence/` 目录下

示例：

```fasta
>protein_1
MKTAYIAKQRQISFVKSHFSRQDILD
>protein_2
GILGYTEHQVVSSDFNSDTHR
```

补充说明：
- 程序会把 FASTA 标题行第一个字段作为 `sequence_id`
- `sequence_read_save.py` 会把非标准氨基酸字符替换为 `-`
- 但下游特征提取更适合标准 20 种天然氨基酸，因此建议输入只包含标准蛋白字符

## 6. 如何进行预测

### 6.1 准备输入文件
把你的输入文件复制到：

```bash
cp /你的路径/your_input.fasta /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code/sequence/
```

### 6.2 运行预测
进入源码目录后执行：

```bash
cd /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code
python DeepSoluE.py -i your_input.fasta -o your_result.csv
```

### 6.3 查看结果
结果文件默认输出到：

```bash
/Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code/results/your_result.csv
```

输出列包括：
- `sequence_id`
- `predicted_probability`
- `result`

其中：
- `result=soluble` 表示预测为可溶
- `result=insoluble` 表示预测为不可溶

## 7. 需要注意的路径硬编码与可微调位置

### 7.1 输入文件路径被写死为 `./sequence/`
以下代码都默认从 `./sequence/文件名` 读取输入，因此你不能直接传系统绝对路径，除非改代码：

- `feature_scripts/sequence_read_save.py`
- `feature_scripts/tmhmm_usearch.py`
- `feature_scripts/Biofea.py`
- `feature_scripts/w2v_kmer_corpus_feature.py`

最小改动方案：
- 不改代码
- 只把 FASTA 文件放进 `sequence/`
- 命令里传文件名，例如 `-i your_input.fasta`

### 7.2 TMHMM/USEARCH 路径是写死的
如你本机安装位置不同，需要修改：

`/Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code/feature_scripts/tmhmm_usearch.py`

重点位置：
- `tmhmm()` 中的 `./softs/tmhmm-2.0c/bin/tmhmm`
- `usearch()` 中的 `./softs/usearch/usearch`
- `usearch()` 中的数据库路径 `./softs/usearch/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta`

### 7.3 模型与缩放器路径也是相对路径
如果你移动了模型目录，需要同步修改：

- `feature_scripts/predict.py`
- `feature_scripts/feature_combine.py`

涉及路径：
- `./model/tf_model/model_1.hdf5` 到 `model_10.hdf5`
- `./model/scaler.gz`
- `./model/w2v/training_k3w2_shffule.model`

### 7.4 必须在源码目录执行
`DeepSoluE.py` 虽然接收 `-i/-o` 参数，但内部所有依赖目录仍按当前工作目录解析，所以必须先：

```bash
cd /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code
```

再运行命令。

## 8. 一套可直接参考的完整部署与预测命令

```bash
conda create -n deepsolue python=3.7.3 -y
conda activate deepsolue
pip install numpy==1.19.2 pandas==1.1.3 biopython==1.78 gensim==3.8.3 scikit-learn==0.23.2 joblib==0.17.0 tensorflow==2.3.0

cd /Volumes/T7_PuppetZ/solu_test/DeepSoluE/DeepSoluE-master_source_code
mkdir -p softs/usearch softs/tmhmm-2.0c/bin
cp /你的下载路径/usearch softs/usearch/
cp /你的下载路径/tmhmm softs/tmhmm-2.0c/bin/
cp /Volumes/T7_PuppetZ/solu_test/DeepSoluE/Ecoli_xray_nmr_pdb_no_nesg_simple_id.fasta softs/usearch/
chmod +x softs/usearch/usearch
chmod +x softs/tmhmm-2.0c/bin/tmhmm

cp /你的路径/your_input.fasta sequence/
python DeepSoluE.py -i your_input.fasta -o your_result.csv
```

## 9. 简要结论
- 环境核心是 Python 3.7 + TensorFlow 2.3 + Biopython + Gensim + Pandas + NumPy
- 还需额外安装 `scikit-learn` 和 `joblib`
- 不强制需要 GPU，CPU 也能预测
- 输入文件必须是蛋白质 `FASTA`
- 最稳妥的运行方式：把 FASTA 放进 `sequence/`，进入 `DeepSoluE-master_source_code` 目录执行预测
- 预测前必须补齐 `TMHMM`、`USEARCH` 和 USEARCH 数据库
