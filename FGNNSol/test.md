# FGNNSol Docker 部署与 `eval_data` 测试记录

本文档基于仓库根目录的 `README.md` 与 `use.md`，并结合当前机器上的实际执行结果整理。目标是：

1. 在 Docker 容器内完成 FGNNSol 环境部署。
2. 不修改作者源代码、训练脚本、预测脚本。
3. 直接抽取作者已提供的 `dataset/eval_data/pkl` 作为测试输入，完成一次可复现预测。
4. 将测试结果保存在 `test_415/` 下。

## 1. 说明与结论

- 仓库作者说明里已经明确：`dataset` 中的 `.pkl` 文件可直接用于训练、测试与预测。
- 本次测试没有重跑完整特征提取链路，而是直接使用 `dataset/eval_data/pkl`，因为这是作者已提供的可直接运行输入，也能避免额外引入 ESM-C 权重、ChimeraX、DSSP 全流程的不确定性。
- 预测脚本 `predict.py` 内部将输入目录固定为 `./dataset/prediction/pkl`，模型路径固定为 `./check_point/best_model/best_model.pt`，并且数据会直接 `.to(torch.device('cuda'))`，因此本次部署按 GPU 方式运行。
- Docker 镜像名/容器名不能含大写字母，所以用户要求中的 `FGNNSol_test` 实际落地为小写 `fgnnsol_test`。这是 Docker 命名规则限制，不是仓库问题。

## 2. 本机基础条件

本次测试依赖以下宿主机条件：

- 已安装 Docker
- 已安装 Conda，路径为 `/home/xuyzh/miniconda3`
- 已配置宿主机 `.condarc`，路径为 `/home/xuyzh/.condarc`
- 宿主机存在可用 NVIDIA GPU
- 当前机器架构为 `linux-aarch64`

说明：

- 由于本机已经部署好 Conda，因此容器内没有重新安装 Conda，而是直接挂载宿主机的 Conda 到容器中使用。
- 这样可以最大程度复用用户现有 Conda 与镜像源配置。

## 3. Docker 文件

仓库根目录新增了 `Dockerfile`，其作用仅是提供一个最小 Ubuntu 22.04 运行壳，不内置 Python 环境，不改作者代码：

```dockerfile
FROM ubuntu:22.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    curl \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/FGNNSol

CMD ["bash"]
```

说明：

- 虽然 `Dockerfile` 中保留了 `PATH=/opt/conda/bin:$PATH`，但本次实际运行时通过 `docker run -e PATH=...` 覆盖为宿主机 Conda 路径。
- 这样做是为了不在镜像构建阶段重复安装 Conda。

## 4. 构建镜像

在仓库根目录执行：

```bash
docker build -t fgnnsol_test .
```

镜像构建完成后，可通过以下命令确认：

```bash
docker images | grep fgnnsol_test
```

## 5. 启动容器

先删除旧容器，再启动新容器：

```bash
docker rm -f fgnnsol_test
```

```bash
docker run -dit --gpus all \
  --name fgnnsol_test \
  -v /home/xuyzh/solu_test/FGNNSol:/workspace/FGNNSol \
  -v /home/xuyzh/miniconda3:/home/xuyzh/miniconda3 \
  -v /home/xuyzh/.condarc:/root/.condarc:ro \
  -e PATH=/home/xuyzh/miniconda3/bin:$PATH \
  fgnnsol_test bash
```

容器状态检查：

```bash
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
```

本次结果为：

- 容器名：`fgnnsol_test`
- 镜像名：`fgnnsol_test`
- 状态：`Up`

## 6. Conda 环境配置

### 6.1 接受 Anaconda ToS

首次在容器内以 root 使用挂载的 Conda 时，需要接受 Anaconda Terms of Service：

```bash
docker exec fgnnsol_test bash -lc \
  'conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
   conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r'
```

### 6.2 创建环境

先创建基础环境：

```bash
docker exec fgnnsol_test bash -lc \
  'conda create -n fgnnsol python=3.9 -y --override-channels -c defaults -c conda-forge'
```

### 6.3 安装运行依赖

根据 README 和 `use.md`，作者建议环境接近：

- Python 3.9
- PyTorch 2.2.2
- torch-geometric 2.6.1
- torch-scatter 2.1.2
- torch-cluster 1.6.3

但由于当前机器是 `linux-aarch64`，且作者给出的旧版本二进制组合在当前渠道下无法直接完整安装，因此做了兼容性升级，实际安装为：

```bash
docker exec fgnnsol_test bash -lc \
  'conda install -n fgnnsol -y --override-channels -c conda-forge \
   python=3.10 pytorch=2.10.* torch-geometric=2.6.1 pandas scikit-learn tqdm numpy'
```

说明：

- 这里将 Python 从 3.9 升级到 3.10。
- PyTorch 从 README 中的 2.2.2 升级为 2.10.0。
- 这样做的原因是 ARM 架构下旧版依赖组合不可解，升级后可正常完成安装并运行预测。
- 本次测试的重点是保持作者代码不改动并跑通现有模型推理，不是完全锁死到 README 中的历史二进制版本。

### 6.4 导出环境包列表

为便于复现，本次环境包列表导出到：

- `test_415/conda_list.txt`

导出命令：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && conda list -n fgnnsol > test_415/conda_list.txt'
```

### 6.5 环境验证

验证 PyTorch、CUDA、PyG：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && \
   /home/xuyzh/miniconda3/bin/conda run -n fgnnsol python -c \
   "import torch, torch_geometric; \
    print(torch.__version__); \
    print(torch.cuda.is_available()); \
    print(torch.cuda.device_count()); \
    print(torch_geometric.__version__)"'
```

本次实际验证结果：

- `torch.__version__ = 2.10.0`
- `torch.cuda.is_available() = True`
- `torch.cuda.device_count() = 1`
- `torch_geometric.__version__ = 2.6.1`

## 7. 不改源码前提下的路径适配

由于作者脚本中存在硬编码路径，本次通过建立符号链接完成适配，不修改任何源码文件。

### 7.1 让 `predict.py` 读取 `eval_data/pkl`

作者预测脚本固定读取：

- `./dataset/prediction/pkl`

但本次测试要求直接抽取：

- `./dataset/eval_data/pkl`

因此在容器内执行：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && \
   mkdir -p dataset/prediction && \
   rm -rf dataset/prediction/pkl && \
   ln -s ../eval_data/pkl dataset/prediction/pkl'
```

说明：

- 这样 `predict.py` 仍然读取它原本的固定路径。
- 实际上该路径已经被重定向到作者提供的 `eval_data/pkl`。

### 7.2 让 `predict.py` 读取实际存在的模型文件

作者 README 中写模型在：

- `./check_point/best_model/best_model.pt`

但仓库实际存在的文件是：

- `./check_point/best_model.pt`

`predict.py` 又恰好固定写成前者，因此通过符号链接适配：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && \
   mkdir -p check_point/best_model && \
   rm -f check_point/best_model/best_model.pt && \
   ln -s ../best_model.pt check_point/best_model/best_model.pt'
```

## 8. ARM 平台兼容处理

### 8.1 问题背景

在 `linux-aarch64` 平台上，按 README 的组合安装时，`torch-scatter` 和 `torch-cluster` 的现成编译包不可用或不可解。

但 `GPSol.py` 与 `torch_geometric.nn.radius_graph` 运行时会依赖这些模块。

### 8.2 处理方式

为保证：

- 不改作者源码
- 仍可完成预测验证

本次在 `test_415/compat/` 下提供了兼容 shim：

- `test_415/compat/torch_scatter/__init__.py`
- `test_415/compat/torch_cluster/__init__.py`

用途：

- `torch_scatter` shim 提供 `scatter_add` 与 `scatter_mean`
- `torch_cluster` shim 提供 `radius_graph`

运行预测时只需在命令前临时追加 `PYTHONPATH`：

```bash
export PYTHONPATH=/workspace/FGNNSol/test_415/compat:$PYTHONPATH
```

说明：

- 这是外部兼容层，不是对作者源码的修改。
- 仅用于本机 ARM 环境下补齐缺失扩展模块。

## 9. 执行预测测试

### 9.1 运行命令

在容器内执行：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && \
   export PYTHONPATH=/workspace/FGNNSol/test_415/compat:$PYTHONPATH && \
   /home/xuyzh/miniconda3/bin/conda run -n fgnnsol python predict.py \
   > test_415/predict.log 2>&1'
```

说明：

- 工作目录必须切换到 `/workspace/FGNNSol`
- 必须设置 `PYTHONPATH` 指向 `test_415/compat`
- 模型和输入数据路径已通过前面的符号链接适配完成

### 9.2 保存预测结果

`predict.py` 默认输出：

- `./predictions.csv`

为满足测试归档要求，本次执行后将其复制到：

- `test_415/predictions_eval_data.csv`

对应命令：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && cp predictions.csv test_415/predictions_eval_data.csv'
```

### 9.3 结果统计

对预测结果做简单统计，输出到：

- `test_415/prediction_summary.txt`

命令：

```bash
docker exec fgnnsol_test bash -lc \
  'cd /workspace/FGNNSol && \
   /home/xuyzh/miniconda3/bin/conda run -n fgnnsol python -c \
   "import pandas as pd; \
    p=pd.read_csv(\"predictions.csv\"); \
    print(p[\"prediction\"].describe().to_string())" \
   > test_415/prediction_summary.txt'
```

## 10. 测试结果

本次 `eval_data/pkl` 目录共成功完成 268 条预测，输出文件如下：

- `test_415/predictions_eval_data.csv`
- `test_415/predict.log`
- `test_415/prediction_summary.txt`
- `test_415/conda_list.txt`

`prediction_summary.txt` 内容如下：

```text
count    268.000000
mean       0.478096
std        0.231102
min        0.112330
25%        0.288071
50%        0.424217
75%        0.673943
max        0.955377
```

`predict.log` 中的关键信息表明预测已完成：

- `data loading...............`
- `data loaded !!!!!!!!!!`
- `Model loaded successfully`
- `Predictions saved to predictions.csv，共268条记录`

## 11. 本次遇到的问题与解决方案

### 问题 1：Docker 名称不能使用 `FGNNSol_test`

现象：

- Docker 镜像名和容器名不能包含大写字母。

解决：

- 统一使用小写 `fgnnsol_test`。

### 问题 2：宿主机 Conda 挂载到其他路径后无法运行

现象：

- 初次尝试将宿主机 Conda 挂载到 `/opt/conda` 时，容器内执行失败。

原因：

- 宿主机 Conda 可执行文件的 shebang 与内部路径写死为 `/home/xuyzh/miniconda3/bin/python`。

解决：

- 将宿主机 Conda 原样挂载到容器内相同绝对路径 `/home/xuyzh/miniconda3`。
- 同时在 `docker run` 中显式设置 `PATH=/home/xuyzh/miniconda3/bin:$PATH`。

### 问题 3：宿主机 `.condarc` 对 ARM 的镜像源返回 403

现象：

- 使用宿主机 `.condarc` 内的部分清华镜像时，`linux-aarch64` 平台请求返回 403。

解决：

- 创建环境和安装关键依赖时使用：
  `--override-channels -c defaults -c conda-forge`

说明：

- 这一步没有删除用户镜像配置，只是在本次容器安装过程中对具体命令做了临时覆盖。

### 问题 4：Anaconda ToS 未接受导致 Conda 无法安装

现象：

- 在容器内第一次用 root 执行 Conda 安装时，Conda 阻止继续。

解决：

- 先执行 `conda tos accept ...` 接受 `pkgs/main` 与 `pkgs/r`。

### 问题 5：README 指定的旧版 PyTorch 组合在 ARM 下不可直接复现

现象：

- `python=3.9 + pytorch=2.2.2 + torch-scatter=2.1.2 + torch-cluster=1.6.3` 在当前 `linux-aarch64` 平台下无法完整求解或缺少可用包。

解决：

- 升级到可安装可运行的组合：
  `python=3.10 + pytorch=2.10.0 + torch-geometric=2.6.1`

说明：

- 保持作者核心代码与推理逻辑不变。
- 这是平台兼容性调整，不是算法修改。

### 问题 6：`torch-scatter` / `torch-cluster` 缺失

现象：

- 预测过程中依赖这两个扩展包，但 ARM 渠道中无法直接获得对应编译版本。

解决：

- 在 `test_415/compat/` 中提供纯 Python 兼容实现。
- 运行时通过 `PYTHONPATH` 注入。

## 12. 需要注意的路径

若在其他机器复现，本次流程中需要替换的路径主要有：

- 宿主机仓库路径：`/home/xuyzh/solu_test/FGNNSol`
- 宿主机 Conda 路径：`/home/xuyzh/miniconda3`
- 宿主机 `.condarc` 路径：`/home/xuyzh/.condarc`
- 容器内工作目录：`/workspace/FGNNSol`

只要这些路径改成目标机器的实际路径即可，其余命令结构可保持不变。

## 13. 最终产物

本次新增或产出的关键文件为：

- `Dockerfile`
- `test.md`
- `test_415/conda_list.txt`
- `test_415/predict.log`
- `test_415/predictions_eval_data.csv`
- `test_415/prediction_summary.txt`
- `test_415/compat/torch_scatter/__init__.py`
- `test_415/compat/torch_cluster/__init__.py`
- `test_415/compat/README.md`

至此，FGNNSol 已在 Docker 容器 `fgnnsol_test` 内完成部署，并基于作者提供的 `dataset/eval_data/pkl` 成功完成一次不改源码的预测测试。
