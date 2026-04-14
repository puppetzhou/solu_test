# GraphSol Docker 测试记录

时间：2026-04-15 01:16:55 HKT

仓库：`/home/xuyzh/solu_test/GraphSol`

结论：已在本机 `linux-aarch64` 上成功构建并运行 GraphSol 的预测流程，容器名为 `graphsol_test`，镜像已保存为：

- `graphsol_test:test415`
- `graphsol_test:run415`

## 1. 作者说明核对

- `GraphSol` 仓库内存在 `README.md`，未找到该仓库自己的 `use.md`。
- 作者原始 `README` 说明：
  - 训练环境测试于 Python 3.7.9、`torch==1.6.0` 等老版本。
  - 预测流程 `Predict/predict.py` 依赖 5 类节点/1 类边特征。
  - 这些特征生成软件并未集成在仓库中，作者要求额外使用 `PSI-BLAST / HH-Suite3 / SPIDER3 / DCA / CCMPred / SPOT-Contact`。
- 由于本机为 arm 架构，且作者锁定环境较老，直接复现 Python 3.7 + torch 1.6 在当前平台不现实，因此按“可运行优先”的原则升级到兼容组合，同时不修改作者源码。

## 2. 容器与环境

### 2.1 Dockerfile

新增文件：

- `test_415/Dockerfile`
- `test_415/condarc`

构建命令：

```bash
cd /home/xuyzh/solu_test/GraphSol
docker build -t graphsol_test:test415 -f test_415/Dockerfile .
```

容器内最终使用的软件组合：

- Python 3.10
- numpy 1.26.4
- pandas 1.5.3
- scikit-learn 1.4.2
- tqdm
- torch 2.2.2

说明：

- 本机现有 `.condarc` 指向清华镜像。
- 但在 Docker 构建阶段，`mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main` 和 `.../cloud/conda-forge` 均返回 `HTTP 403`。
- 因此容器内最终改用官方 `conda-forge` 作为唯一 conda 源。
- 这一步属于“原镜像源在容器网络路径不可用时的兼容调整”。

## 3. 试运行策略

你的要求允许减少 `PLMsol_test.fasta` 的序列数量，并允许额外添加 `fasta -> 特征` 的处理脚本，因此采用以下方案：

1. 从 `/home/xuyzh/solu_test/PLMsol_test.fasta` 中提取一个小子集。
2. 生成 `test_415/graphsol_test.fasta`。
3. 生成与 `Predict/predict.py` 接口兼容的 `Predict/Data/generate/*_oneD.npy` 与 `*_twoD.npy`。
4. 直接调用作者原始脚本 `Predict/predict.py` 完成推理。

这样做的原因：

- 作者完整特征链并不在仓库内。
- 仅凭 fasta 无法直接运行作者原始预测脚本。
- 为证明流程可行，本次额外脚本生成的是“兼容输入特征”，不修改作者源码。

重要说明：

- 本次对 `PLMsol_test.fasta` 的预测属于“流程打通测试”。
- 由于没有接入作者 README 中要求的完整外部特征软件链，当前结果仅用于验证部署与运行可行，不应当当作严格生物学结论使用。

## 4. 新增测试侧脚本

新增文件：

- `test_415/prepare_graphsol_input.py`
- `test_415/run_graphsol_test.sh`

### 4.1 `prepare_graphsol_input.py` 做了什么

- 读取 `/workspace/PLMsol_test.fasta`
- 过滤掉过短、过长或包含非常规字符的序列
- 选取前 5 条合规序列
- 重新命名为 `gs001` 到 `gs005`
- 输出：
  - `test_415/graphsol_test.fasta`
  - `test_415/graphsol_test_mapping.tsv`
- 生成兼容特征：
  - `Predict/Data/generate/gsXXX_oneD.npy`
  - `Predict/Data/generate/gsXXX_twoD.npy`

特征生成逻辑：

- `oneD.npy`
  - 前 23 维使用仓库自带 `BLOSUM62_dim23.txt`
  - 其余维度使用公共统计量填充，保证维度与作者模型一致
- `twoD.npy`
  - 采用作者 `predict.py` 在缺少 `.spotcon` 时使用的近邻带状图思路

### 4.2 运行脚本

```bash
docker rm -f graphsol_test >/dev/null 2>&1 || true

docker run --name graphsol_test \
  --workdir /workspace/GraphSol \
  -v /home/xuyzh/solu_test:/workspace \
  graphsol_test:test415 \
  bash /workspace/GraphSol/test_415/run_graphsol_test.sh
```

脚本内部实际执行：

```bash
cd /workspace/GraphSol
PYTHON_BIN=/opt/conda/envs/graphsol/bin/python
cd /workspace/GraphSol/Predict
"${PYTHON_BIN}" /workspace/GraphSol/test_415/prepare_graphsol_input.py
"${PYTHON_BIN}" predict.py
cp Result/result.csv /workspace/GraphSol/test_415/result.csv
```

## 5. 本次试运行输入

生成的测试 fasta：

- `test_415/graphsol_test.fasta`

ID 映射：

- `test_415/graphsol_test_mapping.tsv`

本次选中的 5 条序列：

- `gs001 -> daad70dcaa114ad41519387830757c7c7307e83b (360 aa)`
- `gs002 -> a128dca033f6ed31d65c87a57f411a57dcec0158 (213 aa)`
- `gs003 -> 708d855b81ee5d7ed06704326b1def579ae8d4d9 (157 aa)`
- `gs004 -> 9eae94b3246f4946f8cfae808400fa556a5520e5 (546 aa)`
- `gs005 -> 115d3147eecb280105c72114810989431309949b (386 aa)`

## 6. 运行结果

结果文件：

- `test_415/result.csv`

结果摘要：

```csv
name,prediction
gs001,0.36821001172065737
gs002,0.633450722694397
gs003,0.2295216828584671
gs004,0.4190455675125122
gs005,0.6149940669536591
```

说明：

- 作者原始 `Predict/predict.py` 已成功执行完成。
- 运行过程中仅出现 pandas 的 `FutureWarning`：
  - `DataFrame.append` 在未来版本将移除。
  - 由于本次固定使用 `pandas==1.5.3`，不影响当前运行。

## 7. 遇到的问题与解决方案

### 问题 1：作者原始环境过旧，不适合当前 arm 平台

现象：

- README 推荐 `Python 3.7.9 + torch 1.6.0`
- 在 `linux-aarch64` 上不适合直接照搬

解决：

- 升级为 `Python 3.10 + torch 2.2.2 + pandas 1.5.3 + numpy 1.26.4`
- 不改作者脚本，仅调整运行环境版本

### 问题 2：容器内 conda 镜像 403

现象：

- 清华镜像在 Docker build 阶段返回 `HTTP 403`

解决：

- 保留问题记录
- 容器内切换到官方 `conda-forge`

### 问题 3：仅有 fasta 无法直接跑作者预测

现象：

- 作者要求外部特征软件链
- 仓库不自带这些软件

解决：

- 按你允许的方式新增预处理脚本
- 直接生成 `Predict/Data/generate` 所需输入矩阵
- 继续调用作者原始 `Predict/predict.py`

### 问题 4：容器初次运行走错 Python

现象：

- 初次执行时调用到了系统 Python，报 `ModuleNotFoundError: No module named 'numpy'`

解决：

- 在 `test_415/run_graphsol_test.sh` 中显式指定：

```bash
PYTHON_BIN=/opt/conda/envs/graphsol/bin/python
```

## 8. GraphSol 使用指南

### 8.1 直接复用当前测试流程

如果只是复现本次流程：

```bash
cd /home/xuyzh/solu_test/GraphSol
docker run --rm \
  --workdir /workspace/GraphSol \
  -v /home/xuyzh/solu_test:/workspace \
  graphsol_test:test415 \
  bash /workspace/GraphSol/test_415/run_graphsol_test.sh
```

输出文件：

- `test_415/graphsol_test.fasta`
- `test_415/graphsol_test_mapping.tsv`
- `test_415/result.csv`

### 8.2 如果要换成新的 fasta

当前测试脚本默认读取：

- `/workspace/PLMsol_test.fasta`

如果要换文件，有两种做法：

1. 直接替换 `/home/xuyzh/solu_test/PLMsol_test.fasta`
2. 修改 `test_415/prepare_graphsol_input.py` 中：

```python
INPUT_FASTA = ROOT / "PLMsol_test.fasta"
```

### 8.3 如果要走作者“严格原始”预测流程

需要额外准备并放到 `Predict/Data/source/` 的文件：

- `.pssm`
- `.hhm`
- `.spd33`
- `.spotcon`
- 以及作者 README 中列出的其他中间特征

然后将输入总 fasta 放到：

- `Predict/Data/upload/input.fasta`

再执行：

```bash
cd /workspace/GraphSol/Predict
/opt/conda/envs/graphsol/bin/python predict.py
```

### 8.4 本次路径变更点

为适配测试，仅变更了输入与输出路径，不改作者源码：

- 输入总 fasta：
  - `Predict/Data/upload/input.fasta`
- 兼容特征目录：
  - `Predict/Data/generate/`
- 测试输出目录：
  - `test_415/`

## 9. 容器转镜像

成功运行后已执行：

```bash
docker commit graphsol_test graphsol_test:run415
```

本地镜像状态：

```bash
graphsol_test:run415
graphsol_test:test415
```

可查看：

```bash
docker images | grep graphsol_test
```
