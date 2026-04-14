# FGNNSol 在 Ubuntu 24.04 上的部署与使用说明

本文档基于当前仓库代码整理，不完全照搬 README，而是以“代码真实要求”为准。目标是完成两件事：

1. 在 Ubuntu 24.04 上创建严格隔离的 `conda` 环境。
2. 明确预测所需输入、目录布局和执行命令。

## 1. 结论先看

FGNNSol 不是“只给一个 fasta 就能直接预测”的程序。按当前代码，预测至少需要以下 5 类输入：

- `predict.csv`
- 每个蛋白一份 `fasta`
- 每个蛋白一份 `pdb`
- 每个蛋白一份 ESM-C 特征 `npy`
- 每个蛋白一份 DSSP 特征 `dssp`
- 每个蛋白一份理化特征 `npy`

然后通过 `feature_extraction/extract_prediction_features.py` 将这些特征整合成 `pkl`，最后再运行 `predict.py` 生成 `predictions.csv`。

另外，当前仓库代码有几个必须提前知道的限制：

- `predict.py` 和 `train.py` 实际按 GPU 写的，`load_data_with_names()` 里直接把数据 `.to(torch.device('cuda'))`，没有纯 CPU 兼容处理。
- 预测特征提取脚本引用的是 `Protein_parameters_train_setting.json`，但仓库实际存在的是 `feature_extraction/Protein_parameters_setting.json`。
- README 中写的模型路径 `./check_point/best_model/best_model.pt` 不对；仓库实际文件是 `./check_point/best_model.pt`，而 `predict.py` 当前却写成了前者。
- 理化特征提取部分的仓库脚本给的是 Windows `.bat`，在 Ubuntu 上不能直接双击运行，需要手工用 `pdb2pqr` 和 ChimeraX 跑同样流程。

## 2. 推荐环境

### 2.1 系统层面依赖

建议先安装这些系统工具：

```bash
sudo apt update
sudo apt install -y git wget curl build-essential
```

如果你要跑 DSSP：

```bash
chmod +x ./feature_extraction/mkdssp.exe
```

如果你要完整跑“理化特征”流程，还需要额外安装：

- `ChimeraX`
- `pdb2pqr`

其中 `ChimeraX` 通常作为独立软件安装，不建议塞进 conda 环境里。

### 2.2 conda 严格隔离环境

以下命令以 Miniconda/Anaconda 已安装为前提。

```bash
conda create -n fgnnsol python=3.9 -y
conda activate fgnnsol

conda config --env --set channel_priority strict

conda install -y -c pytorch -c nvidia \
  pytorch=2.2.2 pytorch-cuda=12.1 torchvision torchaudio

conda install -y -c pyg \
  pyg=2.6.1 pytorch-scatter=2.1.2 pytorch-cluster=1.6.3

conda install -y -c conda-forge \
  numpy=1.24.3 pandas=1.5.3 biopython=1.85 scikit-learn=1.6.1 \
  seaborn=0.13.2 matplotlib-inline=0.1.7 ipython=8.18.1 \
  sentencepiece=0.2.0 transformers=4.37.2 tqdm rdkit

pip install iFeatureOmegaCLI==1.0.2
pip install pdb2pqr
```

说明：

- `torch-geometric` 这里通过 `pyg` 元包安装。
- `rdkit` 建议走 `conda-forge`，不要优先用 pip。
- `iFeatureOmegaCLI` 仓库代码直接 import，因此需要单独安装。
- `pdb2pqr` 不是 README 明写的包，但理化特征流程的 `.bat` 脚本明确依赖它。

### 2.3 ESM-C 依赖

仓库中的 `feature_extraction/extract_ESMC.py` 实际导入的是：

```python
from esm.models.esmc import ESMC
from esm.sdk.api import ESMCInferenceClient, ESMProtein, LogitsConfig, LogitsOutput
```

这说明它依赖的是 EvolutionaryScale 的 ESM-C 代码接口，而不是老的 `fair-esm` 用法。按仓库思路，至少还需要满足下面二选一之一：

1. 安装可用的 `esm` / ESM-C Python 包与模型权重。
2. 按 README 提示，把 ESM-C 代码和模型权重准备好后再运行 `extract_ESMC.py`。

如果这一步没准备好，`dataset/prediction/esmc/*.npy` 无法生成，后续预测也跑不通。

## 3. 建议的初始化命令

在仓库根目录执行：

```bash
conda activate fgnnsol

mkdir -p dataset/prediction/{fasta,pdb,esmc,dssp,extra_feat,pkl}

chmod +x feature_extraction/mkdssp.exe

cp feature_extraction/Protein_parameters_setting.json \
   feature_extraction/Protein_parameters_train_setting.json
```

最后这条 `cp` 很重要，因为代码引用的文件名和仓库现有文件名不一致。

## 4. 输入文件要求

### 4.1 原始表格：`dataset/csvFile/predict.csv`

预测时，`feature_extraction/readme.md` 和 `extract_prediction_features.py` 要求使用：

`dataset/csvFile/predict.csv`

最少应包含以下列：

- `name`
- `sequence`

按当前 `extract_prediction_features.py` 实现，代码还会读取：

```python
dict_pdb_chain = pdb_chain_list.set_index('name')['solubility'].to_dict()
```

所以为了避免脚本报错，预测用的 `predict.csv` 最稳妥也要包含：

- `name`
- `sequence`
- `solubility`

其中 `solubility` 在真实预测时可以填占位值，例如 `0`。

推荐格式：

```csv
name,sequence,solubility
protein1,MSEQUENCEEXAMPLE,0
protein2,ACDEFGHIKLMNPQRSTVWY,0
```

要求：

- `name` 必须唯一。
- `name` 必须与对应的 `.fasta`、`.pdb`、`.npy`、`.dssp` 文件基名完全一致。
- `sequence` 必须是标准蛋白单字母序列，建议只包含 `ACDEFGHIKLMNPQRSTVWY`。

### 4.2 FASTA 输入

预测流程需要单蛋白 FASTA 文件，位置：

`dataset/prediction/fasta/`

按 `feature_extraction/csv_to_fasta.py` 和 `extract_prediction_features.py`，预测数据的 FASTA 文件名应为：

`{name}.fasta`

而不是训练集里常见的：

`{name}-model_v4.fasta`

每个 fasta 建议内容如下：

```fasta
>protein1
MSEQUENCEEXAMPLE
```

仓库已提供批量转换脚本：

```bash
cd feature_extraction
python csv_to_fasta.py
cd ..
```

### 4.3 PDB 输入

预测结构文件位置：

`dataset/prediction/pdb/{name}.pdb`

代码真实要求来自 `feature_extraction/process_structure.py`：

- 只解析 `ATOM` 记录。
- 不读取 `HETATM`。
- 只使用每个残基的 `N`、`CA`、`C`、`O` 和侧链原子坐标。
- 如果残基缺失 `O`，代码会自动用 `CA` 顶替。
- 如果残基没有侧链原子，代码会自动用 `CA` 顶替。

这意味着：

- 如果 PDB 中有水分子，通常写在 `HETATM` 里，`get_pdb_xyz()` 不会读取它们。
- 但为了避免 DSSP、PDB2PQR、ChimeraX 等外部工具受干扰，仍然建议只保留目标蛋白链，去掉水分子、离子、小分子配体和无关链。
- 最稳妥的输入是单链、单蛋白、残基编号连续且与 `sequence` 一致的结构。

PDB 建议满足：

- 来自 AlphaFold/AlphaFold3 或其他单蛋白结构预测结果。
- 文件基名必须等于 `name`。
- 链尽量只保留 1 条。
- 不要混入多个蛋白或复合物。

## 5. 预测前需要生成的中间特征

最终 `predict.py` 不是直接读 fasta/pdb，而是读：

`dataset/prediction/pkl/*.pkl`

这些 `pkl` 由以下几类特征拼出来：

### 5.1 ESM-C 特征

输出目录：

`dataset/prediction/esmc/{name}.npy`

运行示例：

```bash
cd feature_extraction
python extract_ESMC.py
cd ..
```

前提：

- 你已经把 ESM-C 依赖和模型权重准备好。

### 5.2 DSSP 特征

输出目录：

`dataset/prediction/dssp/{name}.dssp`

运行：

```bash
cd feature_extraction
python extract_DSSP.py
cd ..
```

前提：

- `feature_extraction/mkdssp.exe` 可执行。

### 5.3 理化特征

输出目录：

`dataset/prediction/extra_feat/{name}.npy`

这部分流程由 3 段组成：

1. `pdb -> pqr`
2. `pqr -> txt`（通过 ChimeraX）
3. `txt/csv -> npy`

仓库给的是 Windows `.bat`，Ubuntu 24.04 上需要手工执行同样逻辑。

#### 第一步：PDB 转 PQR

在 `feature_extraction/extract_extra_features/` 目录下，为所有预测 PDB 执行：

```bash
mkdir -p feature_extraction/extract_extra_features/prediction_pqr

for f in dataset/prediction/pdb/*.pdb; do
  base=$(basename "$f" .pdb)
  pdb2pqr --ff=CHARMM "$f" \
    "feature_extraction/extract_extra_features/prediction_pqr/${base}.pqr"
done
```

#### 第二步：用 ChimeraX 生成日志

进入目录：

```bash
cd feature_extraction/extract_extra_features
```

在 ChimeraX 命令行运行：

```text
runscript prediction_pqr_chimera.py
pqr_processor
```

这一步会生成：

`feature_extraction/extract_extra_features/prediction_txt/*.txt`

#### 第三步：汇总并转为 npy

仍在 `feature_extraction/extract_extra_features/` 下执行：

```bash
python prediction_extract_extra_feat.sh
```

如果系统把 `.sh` 当文本而不是 Python 脚本运行，正确命令应为：

```bash
bash prediction_extract_extra_feat.sh
```

注意这里脚本里实际依次调用的是：

- `prediction_construction_features.py`
- `prediction_add_feature1.py`
- `prediction_csv_to_npy.py`

完成后会得到：

`dataset/prediction/extra_feat/{name}.npy`

## 6. 生成预测用 PKL

全部中间特征准备完成后，在仓库根目录运行：

```bash
cd feature_extraction
python extract_prediction_features.py
cd ..
```

成功后会生成：

`dataset/prediction/pkl/{name}.pkl`

这一步依赖以下文件都已就位：

- `dataset/csvFile/predict.csv`
- `dataset/prediction/fasta/{name}.fasta`
- `dataset/prediction/pdb/{name}.pdb`
- `dataset/prediction/esmc/{name}.npy`
- `dataset/prediction/dssp/{name}.dssp`
- `dataset/prediction/extra_feat/{name}.npy`

## 7. 执行预测

理论上根目录命令就是：

```bash
python predict.py
```

输出文件：

`predictions.csv`

格式为：

```csv
protein,prediction
protein1,0.7345
protein2,0.1821
```

## 8. 但当前仓库有两个必须先修正的问题

如果完全按当前代码直接运行，预测命令大概率会失败，因为：

### 8.1 模型路径写错了

`predict.py` 当前写的是：

```python
model.load_state_dict(torch.load("./check_point/best_model/best_model.pt", map_location=device))
```

但仓库实际文件是：

`check_point/best_model.pt`

所以实际应改为：

```python
model.load_state_dict(torch.load("./check_point/best_model.pt", map_location=device))
```

### 8.2 代码实际强依赖 CUDA

虽然脚本里写了：

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

但前面又直接执行了：

```python
data = pickle.load(f).to(torch.device('cuda'))
```

以及：

```python
y_hat = torch.tensor([]).cuda()
```

所以当前 `predict.py` 实际不能作为纯 CPU 脚本使用。部署时应默认按“需要 NVIDIA GPU + CUDA 可用”准备。

## 9. 最小可执行流程

如果你已经准备好了 ESM-C、DSSP、PDB2PQR、ChimeraX，并按上面要求放好了输入文件，那么最小执行顺序是：

```bash
conda activate fgnnsol

mkdir -p dataset/prediction/{fasta,pdb,esmc,dssp,extra_feat,pkl}

cp feature_extraction/Protein_parameters_setting.json \
   feature_extraction/Protein_parameters_train_setting.json

cd feature_extraction
python csv_to_fasta.py
python extract_ESMC.py
python extract_DSSP.py
cd extract_extra_features
bash prediction_extract_extra_feat.sh
cd ..
python extract_prediction_features.py
cd ..
python predict.py
```

其中真正最耗时、最容易出问题的是：

- ESM-C 模型和权重准备
- PDB2PQR
- ChimeraX
- `predict.py` 的模型路径与 CUDA 假设

## 10. 输入格式的最终回答

针对“input 为什么格式（fasta or pdb，pdb 是否需要去除水分子等）”，结论如下：

- 不是二选一，预测需要 `fasta + pdb + csv`，并且还要继续生成 `esmc/dssp/extra_feat` 中间文件。
- `csv` 是入口索引文件，至少要有 `name` 和 `sequence`，按当前代码最好再加一列占位 `solubility`。
- `fasta` 必须是单蛋白序列文件，文件名为 `{name}.fasta`。
- `pdb` 必须是对应同一蛋白的结构文件，文件名为 `{name}.pdb`。
- 从代码看，水分子通常不会被核心 PDB 解析逻辑读取，因为它只处理 `ATOM` 行。
- 但为了 DSSP、PDB2PQR、ChimeraX 稳定，仍然强烈建议先去掉水分子、离子、配体、无关链，只保留目标蛋白链。

## 11. 建议

如果你的目标是“先把仓库跑通做一次测试”，建议优先采用下面策略：

1. 先不要自己重新生成全部训练特征。
2. 只做预测流程。
3. 先用 1 到 2 个蛋白做最小测试集。
4. 先修正 `predict.py` 的 checkpoint 路径。
5. 默认在有 CUDA 的 GPU 机器上跑。

