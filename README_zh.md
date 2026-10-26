# rebot_grasp 详细安装与使用说明

[项目首页](README.md) · [B 站实机演示](https://www.bilibili.com/video/BV1iZH36UEpJ/)

本项目在 Seeed-Projects/reBot-DevArm-Grasp 基础上增加 VLM + SAM + GraspNet 自然语言抓取、容器几何放置和连续任务。下文保留基础流程的安装教程，并补充新增功能；新增功能概览见项目首页。

<p align="center">
  <img src="https://raw.githubusercontent.com/Seeed-Projects/reBot-DevArm/main/media/v1.0.png" alt="reBot Arm B601" />
</p>

<p align="center">
    <a href="./LICENSE">
        <img src="https://img.shields.io/badge/Original_contributions-MIT-blue.svg" alt="Original contributions: MIT" />
    </a>
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
    <img src="https://img.shields.io/badge/Platform-Ubuntu%2022.04+-orange.svg" alt="Platform" />
    <img src="https://img.shields.io/badge/Camera-RGB--D-green.svg" alt="Camera" />
    <img src="https://img.shields.io/badge/Detection-YOLO-yellow.svg" alt="YOLO" />
</p>

<p align="center">
  <strong>深度感知 · 目标检测 · 手眼标定 · 自主抓取 · 全开源</strong>
</p>

<p align="center">
  <strong>
    <a href="./README_zh.md">简体中文</a> &nbsp;|&nbsp;
    <a href="./docs/UPSTREAM_GUIDE_EN.md">上游英文教程</a>
  </strong>
</p>

---

##  项目介绍

**reBot Arm B601 视觉夹取 Demo** 是基于 [reBot Arm B601](https://github.com/vectorBH6/reBotArm_control_py) 机械臂控制库与 RGB-D 深度相机的视觉抓取算法演示项目。系统支持 B601 的 DM 与 RS 两种机械臂配置，通过 YOLO 模型实时识别桌面物体，利用 OBB 最小外接矩形估计夹取姿态，经手眼标定将相机坐标系下的抓取点变换到机械臂基坐标系，最终驱动机械臂完成自主抓取。

###  核心功能

- 📷 **深度感知** — 支持 Orbbec Gemini 2 与 RealSense D435i / D405 等 RGB-D 深度相机
- 🔍 **目标检测** — 基于 YOLO 模型识别，支持开放词汇自定义类别
- 📐 **姿态估计** — OBB 最小外接矩形短轴方向估计夹爪朝向，深度分位数估计抓取高度
- 🔄 **坐标变换** — TSAI 手眼标定（Eye-in-Hand），将相机系抓取点变换到机械臂基坐标系
- 🦾 **运动执行** — reBotArm_control_py IK + 轨迹控制器，内置夹爪力控状态机

---

## 硬件配置

| 组件 | 型号 / 要求 |
|------|------------|
| 机械臂 | reBot Arm B601（DM / RS 两种配置） |
| 深度相机 | Orbbec Gemini 2、Intel RealSense D435i / D405 |
| 通信接口 | USB2CAN 串口桥接器（机械臂）；USB 3.0（相机） |
| 主机 | Ubuntu 22.04+，Python 3.10，x86_64 |

**接线说明**

1. 将深度相机通过 USB 3.0 连接到主机
2. 将 USB2CAN 适配器连接到机械臂 CAN 总线并插入主机 USB 口
3. 配置设备权限：

```bash
sudo chmod a+rw /dev/bus/usb/*/*   # 深度相机 USB 权限
sudo chmod 666 /dev/ttyUSB0        # USB2CAN（端口号按实际调整）
```

---

## 环境安装

### Step 1. 克隆仓库

克隆本项目（包含 VLM、SAM、容器几何规划和连续任务扩展）：

```bash
git clone https://github.com/goatwzz/rebot_grasp.git
cd rebot_grasp
```

基础代码来自 [Seeed-Projects/reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)，原始开发仓库为 [EclipseaHime017/reBot-DevArm-Grasp](https://github.com/EclipseaHime017/reBot-DevArm-Grasp)。授权范围见 [第三方说明](THIRD_PARTY_NOTICES.md)。

### Step 2. 创建并配置 conda 环境

```bash
conda env create -f environment.yml
conda activate rebotarm
```
:::tip
如果你想使用其他环境名，可以将命令中的 `rebotarm` 替换为自定义名称。

:::


### Step 3. 安装机械臂控制库

```bash
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
cd sdk/reBotArm_control_py
pip install -e .
cd ../..
```

如果 `pip install -e .` 报 `Multiple top-level packages discovered in a flat-layout`，请在 `reBotArm_control_py` 的 `pyproject.toml` 中加入显式包发现配置，然后重新执行 `pip install -e .`：

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["reBotArm_control_py*"]
```

视觉抓取程序会读取该 SDK 配置，并自动选择对应的机械臂控制模式与夹爪参数。

### Step 4. 安装深度相机 SDK

本项目支持 Orbbec Gemini 2 与 RealSense D435i / D405 等 RGB-D 深度相机。请根据实际使用的相机安装对应 SDK；如果当前环境已经能正常导入相机驱动，可跳过本步骤。

**Orbbec Gemini 2**

Orbbec Gemini 2 依赖 **pyorbbecsdk**（Orbbec SDK v2 的 Python 版本）。优先推荐直接安装预编译 Python 包：

**方式一：通过 pip 安装（推荐）**

```bash
pip install pyorbbecsdk2
```

**方式二：从 GitHub 获取**

```bash
# 安装编译依赖
sudo apt-get install -y cmake build-essential libusb-1.0-0-dev

cd sdk
git clone https://github.com/orbbec/pyorbbecsdk.git
cd pyorbbecsdk
pip install -e .
```

对于中国大陆用户可以使用
```bash
git clone https://gitee.com/orbbecdeveloper/pyorbbecsdk.git
```

源码安装时，请先通过 CMake 编译生成原生扩展，确保 `install/lib` 中已有 `pyorbbecsdk*.so` 和 Orbbec 动态库，再执行 `pip install -e .`。

注意，如果上述安装过程中均发生错误导致安装失败，请参考下方Orbbec官方文档进行安装操作。

:::tip
首次使用建议安装 udev 规则：

```bash
sudo bash scripts/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

**验证安装**

```bash
python -c "import pyorbbecsdk; print('pyorbbecsdk OK')"
```
:::

**OrbbecViewer（可选，用于验证相机）**

下载预编译包后运行 `OrbbecViewer`，可在运行 Demo 前确认相机连接和深度流正常。

- GitHub：https://github.com/orbbec/OrbbecSDK_v2/releases
- Gitee：https://gitee.com/orbbecdeveloper/OrbbecSDK_v2/releases

**RealSense D435i / D405**

RealSense 相机依赖 `pyrealsense2`。通常可以直接通过 pip 安装：

```bash
pip install pyrealsense2
python -c "import pyrealsense2; print('pyrealsense2 OK')"
```

如果系统需要完整的 RealSense 工具链或 udev 规则，请参考 RealSense SDK 官方文档安装 `librealsense2`。


**SDK 资料汇总**

| 资料 | 链接 |
|------|------|
| Gemini 2 产品页 | https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51 |
| 开发资料总链接 | https://www.orbbec.com.cn/index/Download2025/info.html?cate=121&id=1 |
| Orbbec SDK v2 | https://github.com/orbbec/OrbbecSDK_v2 |
| SDK v2 API 文档 | https://orbbec.github.io/docs/OrbbecSDKv2_API_User_Guide/ |
| pyorbbecsdk | https://github.com/orbbec/pyorbbecsdk |
| pyorbbecsdk 文档 | https://orbbec.github.io/pyorbbecsdk/index.html |
| ROS2 Wrapper | https://github.com/orbbec/OrbbecSDK_ROS2/tree/v2-main |
| Intel RealSense SDK | https://github.com/realsenseai/librealsense |

### Step 5. 配置 GraspNet（可选）

为了实现对物体夹取姿态更准确的估计，本项目对[graspnet-baseline](https://github.com/graspnet/graspnet-baseline)进行了适配，从而提升机械臂夹取的性能。

GraspNet 的 `pointnet2` / `knn` 扩展需要 CUDA 编译器。开始前先确认当前环境可以找到 `nvcc`，并检查 `nvcc` 的 CUDA 版本是否和 PyTorch 编译时使用的 CUDA 版本一致：

```bash
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

第一次配置本环境时，`environment.yml` 中的 `torchaudio` 通常已经把 PyTorch 一起装好了；如果没有装好（或装成了 CPU 版），请按下面步骤显式安装并验证 CUDA 版 PyTorch。GraspNet 的 `pointnet2` / `knn` 扩展依赖 PyTorch 和与之版本一致的 CUDA 编译器，全部完成后再开始编译。

**第 1 步：确认显卡**

```bash
nvidia-smi
```

能列出 NVIDIA 显卡再继续；没有 NVIDIA 显卡只能使用 CPU 版 PyTorch，无法编译和运行 GraspNet 的 CUDA 算子。

**第 2 步：检查 / 安装 PyTorch**

先检查当前环境中是否已有 PyTorch：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

- 已安装且 `torch.cuda.is_available()` 输出 `True`：跳过安装，直接到第 3 步。
- 报 `ModuleNotFoundError`（未安装），或 `torch.version.cuda` 为 `None`（装成了 CPU 版）：按下面的命令安装 CUDA 版。以 CUDA 13.0 为例：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

`--index-url` 中的 CUDA 版本按显卡选择：

| 显卡 | 推荐的 PyTorch CUDA 构建 |
| --- | --- |
| RTX 50 系（Blackwell，如 RTX 5070 / 5090） | `cu128` 或更新（如 `cu130`） |
| RTX 40 / 30 系（Ada / Ampere） | `cu121` 或 `cu118` |
| 无 NVIDIA 显卡 | `cpu`（仅 CPU，不能跑 GraspNet） |

**第 3 步：安装与 PyTorch 匹配的 CUDA 编译器**

`pointnet2` / `knn` 是本地 CUDA 扩展，编译时 `nvcc --version` 显示的 CUDA 版本必须与 `torch.version.cuda` 一致，否则会报 `The detected CUDA version (...) mismatches the version that was used to compile PyTorch (...)`。先查看两个版本：

```bash
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

以 `torch.version.cuda` 显示 `13.0` 为例，在 conda 环境中安装匹配的编译器：

```bash
conda install -c nvidia cuda-nvcc=13.0 cuda-cudart-dev=13.0

which nvcc       # 应指向 $CONDA_PREFIX/bin/nvcc，而不是 /usr/bin/nvcc
nvcc --version   # 应与 torch.version.cuda 一致
```

`cuda-nvcc` 提供编译器，`cuda-cudart-dev` 提供 `targets/x86_64-linux/include` 下的 CUDA 头文件，两者都需要。conda 环境激活后 `$CONDA_PREFIX/bin` 位于 `PATH` 最前，会自动遮蔽系统自带的旧版 `nvcc`（例如 Ubuntu 仓库安装的 CUDA 11.5），无需卸载系统 CUDA。

> **Warning**
> 不建议用降级 PyTorch 的方式去迁就系统旧 `nvcc`：旧 CUDA 不一定支持你的显卡算力（CUDA 11.5 最高只支持 sm_86；RTX 40 系需要 CUDA 11.8+，RTX 50 系需要 CUDA 12.8+），且旧版 PyTorch 与本项目其他依赖（如新版 ultralytics）可能不兼容。

**第 4 步：验证**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
nvcc --version
```

两者 CUDA 版本一致、且 `torch.cuda.is_available()` 输出 `True`，即完成 PyTorch 配置，可以继续编译 CUDA 算子。

```bash
cd sdk
git clone https://github.com/graspnet/graspnet-baseline.git
cd graspnet-baseline

# 按你的 CUDA 版本安装 PyTorch 后，再安装 GraspNet 运行依赖
pip install open3d tensorboard Pillow tqdm

# 编译本地算子前配置 CUDA 编译路径。
export CUDA_HOME=$CONDA_PREFIX
export TORCH_CUDA_ARCH_LIST="12.0"
export CPATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPATH
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cu13/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 编译 CUDA 算子
cd pointnet2
pip install . --no-build-isolation
cd ../knn
pip install . --no-build-isolation
cd ..

# 安装 GraspNet API
git clone https://github.com/graspnet/graspnetAPI.git
cd graspnetAPI
sed -i "s/'sklearn'/'scikit-learn'/" setup.py
pip install .
cd ../../..
```

:::tip
注：如果直接参考graspnet-baseline官方仓库文档使用 `python setup.py install` 可能报 CUDA / PyTorch 相关错误，建议使用 `pip install . --no-build-isolation`，让扩展在当前 conda 环境中复用已安装的 PyTorch 与 CUDA 配置进行编译。
:::

:::tip
如果编译时报 `fatal error: cusparse.h: No such file or directory`，先运行 `find $CONDA_PREFIX -name cusparse.h`，并把包含 `cusparse.h` 的目录加入 `CPATH` / `CPLUS_INCLUDE_PATH`。如果 CUDA 头文件来自 conda `cuda-toolkit`，路径通常是 `$CONDA_PREFIX/targets/x86_64-linux/include`，而不是上面示例里的 pip `nvidia/cu13/include` 路径。
:::

:::tip
此外，GraspNet API 的依赖中可能仍使用 `sklearn` 包名。上面的 `sed` 命令会将 `sklearn` 替换为 `scikit-learn`，避免安装时出现包名提示。除非同步调整 GraspNet API 的依赖栈，否则建议保留其 `numpy==1.23.4` 约束，因为 `transforms3d==0.3.1` 仍使用 `np.float` 等 NumPy 别名。
:::

**配置预训练模型**

在graspnet-baseline 官方仓库下载 GraspNet 官方预训练权重[Google](https://drive.google.com/file/d/1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk/view)、[Baidu](https://pan.baidu.com/s/1Eme60l39tTZrilF0I86R5A)，将 下载好的`checkpoint-rs.tar` 放到：

```bash
sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar
```

然后在 `config/default.yaml` 中确认：

```yaml
graspnet:
  checkpoint: "checkpoint-rs.tar"
```

`checkpoint` 支持三种写法：仅文件名会自动从 `sdk/graspnet-baseline/checkpoints/` 查找；相对路径会按项目根目录解析；绝对路径会直接使用。

---

## 目录结构

```
rebot_grasp/
├── config/
│   ├── default.yaml              # 主配置文件
│   └── calibration/
│       └── <camera_type>/
│           ├── intrinsics.npz    # 相机内参
│           └── hand_eye.npz      # 手眼标定结果
├── drivers/
│   ├── camera/
│   │   ├── base.py               # 相机抽象基类
│   │   ├── orbbec_gemini2.py     # Gemini 2 驱动
│   │   └── realsense.py          # RealSense 驱动（备用）
│   └── robot/
│       └── grasp_driver.py       # 基于机械臂 SDK 的轻量抓取辅助
├── calibration/
│   ├── aruco_pose.py             # ArUco 位姿估计
│   └── hand_eye.py               # 手眼标定求解
├── utils/
│   ├── ordinary_grasp.py         # OBB 抓取姿态估计与可视化
│   ├── place_utils.py            # set.py 与 GraspNet 共用的盒子定位和放置动作
│   ├── placement_geometry.py      # VLM 容器三维几何与模拟放置预览
│   └── transforms.py             # 坐标变换工具
├── scripts/
│   ├── main.py                   # 主抓取程序
│   ├── set.py                    # 抓取与放置程序
│   ├── grasp.py                  # GraspNet 抓取后退回准备位，保持夹持
│   ├── grasp_set.py              # GraspNet 抓取后放入盒子
│   ├── ordinary_grasp_pipeline.py
│   ├── object_detection.py
│   └── collect_handeye_eih.py
├── sdk/
│   ├── pyorbbecsdk/              # Orbbec SDK Python 封装
│   └── reBotArm_control_py/      # reBot Arm SDK
└── environment.yml               # 推荐的 conda 环境文件
```


## 运行与调试

### 0. 确认机械臂版本与 SDK 配置

运行会连接机械臂的脚本前，请先确认机械臂版本、电源和 SDK 配置一致：

- 请先完成对应机械臂的基础准备：[B601-DM 快速入门](https://wiki.seeedstudio.com/cn/rebot_b601_dm_getting_started/) 或 [B601-RS 快速入门](https://wiki.seeedstudio.com/cn/rebot_b601_rs_getting_started/)。
- 在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中选择对应的硬件配置：

```yaml
hardware_yaml: rebotarm_dm.yaml
```

或：

```yaml
hardware_yaml: rebotarm_rs.yaml
```

- B601-DM 使用 24V DC 电源，B601-RS 使用 48V DC 电源，请确认电源适配器和接线与机械臂版本一致。
- 使用 B601-DM 时，请确认 SDK 配置中的串口桥接器设备路径与实际设备一致。
- 使用 B601-RS 时，运行标定或抓取脚本前请先启动 CAN 接口：

```bash
sudo ip link set can0 down 2>/dev/null
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up
ip -details link show can0
```

也可在仓库根目录手动运行接口初始化脚本：

```bash
bash scripts/setup_can.sh
```

脚本在 `can0` 已启动且 bitrate 为 `1000000` 时沿用配置；否则配置接口，需要 sudo 时提示授权。源码中的 `config/conda/rebotarm_can_activate.sh` 仅为可选自动激活钩子，使用前需按自己的仓库路径配置，不会在克隆或创建环境时自动安装。

### 1. 手眼标定（抓取前必做）

```bash
python scripts/collect_handeye_eih.py
```

自动模式下，机械臂会自动遍历 50 个预设位姿，检测到 ArUco 稳定后自动采样。正常结束或中途打断时，脚本都会尝试计算并保存标定结果；至少需要 5 个样本，建议 ≥15 个样本以获得更稳的结果。

如需手动推动机械臂采集，可使用：

```bash
python scripts/collect_handeye_eih.py --manual
```

手动模式下，机械臂会进入重力补偿状态。将末端推到合适视角后按 `Enter` 采集，按 `c` 或 `q` 结束并计算。

:::tip
如果您在校准之后发现机械臂的抓取精度无法满足需求，可以设置`config/default.yaml`参数 `calibration.hand_eye_compensation_m`中的 `X（前后）、Y（左右）、Z（高低）` 参数给予位置补偿。
:::

### 2. `scripts/main.py` — 主抓取程序

完整的视觉抓取流水线：

1. 初始化 RGB-D 相机，确认图像流可用
2. 机械臂与夹爪使能，移动到预备高位
3. 实时相机预览 + YOLO 目标检测与实例分割
4. OBB 短轴估计夹爪朝向，深度分位数估计抓取高度
5. 按 `G` 冻结帧，经手眼变换计算机械臂目标位姿
6. 机械臂移动到预抓取点 → 下降 → 夹爪闭合 → 提升 → 回预备位

### 3. `scripts/set.py` — 抓取与放置程序

功能效果：将香蕉抓取并放置到盒子里面

完成流程：
1. 相机与机械臂初始化，移动到预备点位
2. 实时相机预览 + YOLO 目标检测与实例分割
3. 按 `G` 冻结帧，经手眼变换计算机械臂目标位姿
4. 机械臂移动抓取香蕉并抬高
5. 机械臂将香蕉放置在盒子内，并回归初始姿态
6. 按 `Q` 退出系统，机械臂回归零点


### 4. `scripts/ordinary_grasp_pipeline.py` — 简化抓取测试

不依赖机械臂，仅验证 OBB 抓取姿态估计和可视化效果，适合调试感知模块。

### 5. `scripts/graspnet_camera_demo.py` — GraspNet 相机估计 Demo

不连接机械臂，仅使用 RGB-D 相机运行 GraspNet 6D 夹取姿态估计。脚本会保留实时相机预览，并使用 YOLO 检测框选择目标区域，再从 GraspNet 全场景候选中筛选目标 bbox 内的可行夹取。按 `G` 或 `Space` 对当前帧推理，按 `R` 恢复实时预览，按 `Q` 或 `Esc` 退出；推理后可通过 Open3D 查看点云与夹取候选。

```bash
python scripts/graspnet_camera_demo.py
```

### 6. `scripts/grasp.py` — GraspNet 机械臂抓取程序

基于 `graspnet_camera_demo.py` 的估计结果接入机械臂执行流程：YOLO 选择目标，GraspNet 输出 6D 夹取姿态，经手眼标定转换到机械臂基坐标系，再检查 IK 可达性并执行预夹取、夹取、退回动作。调试时建议先使用 `--dry-run` 只打印目标位姿和候选筛选结果。

```bash
python scripts/grasp.py --dry-run
python scripts/grasp.py --target-class "light blue coffee cup"
```

抓香蕉并返回准备位、保持夹持：

```bash
python scripts/grasp.py --target-class "yellow banana"
```

#### GraspNet 抓取并放入盒子：`scripts/grasp_set.py`

使用 GraspNet 计算香蕉抓取姿态，盒子的识别、深度分位数、中心定位和放置动作与 `scripts/set.py` 共用。保持原放置参数：盒子中心的基座坐标 XY、固定基座 Z=`0.08 m`、RPY=`(0, 1.2, 0)`，到位后张开夹爪，延时 `0.8 s` 收爪并返回准备位。抓取后直接抬到准备位，再去盒子位置，与 `set.py` 的顺序相同，不额外执行 `grasp.py` 的沿工具轴退回步骤。

```bash
python scripts/grasp_set.py --target-class "yellow banana"
python scripts/grasp_set.py --target-class "yellow banana" --dry-run
```

默认目标为 `yellow banana`，按 `G` 或 `Space` 执行，`R` 恢复预览。香蕉和盒子需要同时可见，执行时保持盒子不动。没有有效盒子、放置位不可达时不会开始抓取；抓空或中途运动失败会停止后续放置。两个入口均默认读取 `config/default.yaml`，也可以用 `--config` 指定已有的其他 YAML 文件，抓取补偿和插入深度继续从所选配置读取。

`main.py`、`grasp.py`、`set.py` 和 `grasp_set.py` 按 `Q` 或 `Esc` 退出时，夹爪已在零位则不动作；未归零且未检测到持物状态时回零，不再为退出额外张开夹爪。

注意：这两个 GraspNet 机械臂脚本的 `--dry-run` 仅跳过抓取和放置动作，启动时仍会使能并移动到准备位，退出时仍会回零。纯相机预览请使用 `scripts/graspnet_camera_demo.py`。

### 7. `scripts/object_detection.py` — 基础检测 Demo

纯 YOLO 检测演示，实时显示检测框和置信度，无抓取逻辑。

### 8. `scripts/vlm_target_demo.py` — 第一阶段：文字＋图像理解

输入中文自然语言，例如“把左边那根黄色香蕉放进盒子”，程序将指令和一张相机快照发送给视觉语言模型，在同一次请求中返回抓取目标、放置目标各自的名称、二维框和选择理由。窗口中绿色 `PICK TARGET` 为抓取物体，橙色 `PLACE TARGET` 为放置目标；小图上简写为 `PICK` 和 `PLACE`。中文名称与理由显示在终端。红点只是抓取目标的二维框中心，尚不是 GraspNet 抓取点。本阶段不连接机械臂，不执行准备位、抓取或放置，也不加载 YOLO、SAM、GraspNet 或语音模型。

#### 必须补充的服务配置

服务配置位于 `config/default.yaml` 的 `vlm` 段：

- `base_url`：当前配置为 `https://1pkapi.com/v1`，不要填写 `/dashboard` 控制台地址。如使用其他兼容 chat-completions 的服务，需要改成对应地址，并更换该服务的密钥。
- `model`：当前配置为 `gpt-6.1-sol`（平台正式模型 ID，需保留连字符）。更换时需使用账号可调用、支持图像输入和目标定位的模型；具体能力以实际图片请求验证结果为准。
- `api_key_env`：优先读取环境变量 `VLM_API_KEY`，不要将真实密钥写进 YAML 或源码。
- `api_key_file`：没有环境变量时，读取项目根目录下的 `config/vlm_api_key.local`。本地文件只保存密钥文本，权限设为 `600`，已加入 Git 忽略规则；文件不会随代码分发，换机器时需重新配置。

也可以通过环境变量 `VLM_BASE_URL`、`VLM_MODEL` 覆盖地址和模型。只有图像和指令发往配置的远程服务，本阶段没有离线大模型推理。

```bash
cd rebot_grasp  # 从仓库的父目录进入项目
conda activate rebotarm

# 新环境需要自行配置密钥；在当前终端输入，输入时不回显。
# 同时按服务商要求配置 vlm.base_url / vlm.model，或 VLM_BASE_URL / VLM_MODEL。
read -rsp "VLM API Key: " VLM_API_KEY
export VLM_API_KEY
printf '\n'

# 仅检查配置，不打开相机、不请求服务。
python scripts/vlm_target_demo.py --check-config

# 开相机预览，在终端输入中文指令并回车。
python scripts/vlm_target_demo.py
```

也可以直接给一条指令，启动相机后自动识别一张快照：

```bash
python scripts/vlm_target_demo.py --instruction "抓左边的香蕉"
```

窗口内按 `G` 或空格用上一条指令识别新快照，`R` 恢复实时预览并丢弃旧请求的待返回结果，`Q/Esc` 退出。终端也支持输入 `g`、`r`、`q` 后回车，分别重试、恢复预览、退出；这些控制命令不会作为自然语言指令发送给模型。`/quit` 也可退出。网络请求在后台线程执行，返回的框只画在本轮快照上。按 `R` 不会取消已发送的网络请求，待请求结束后才能开始下一次识别。

每轮结果（或错误）输出完成后，终端会重新显示“请输入下一条中文指令”。可以直接输入新指令，无需先按 `R`：程序会获取一张新快照、清除旧框，再显示本轮的新目标框。等待下一条指令时保留上一轮标注，方便观察。

交互预览在启动、每次新提示和退出时恢复终端回显状态，避免窗口 `Q/Esc` 退出时后台 `readline` 仍在等待输入，留下无回显的终端。如果之前的旧进程已导致“打字看不见，但回车会执行”，先退出程序，再输入 `stty sane` 并回车恢复终端，然后重新启动。第一、二阶段共用此修复；中文退格编辑继续使用 `readline`。

如果出现 HTTP `402`、服务错误码 `30001` 或 `account balance is insufficient`，表示模型服务的账户余额不足，需要在服务商控制台检查余额和充值；程序会显示具体错误，不自动重复请求。

没有相机时可用已有图片验证服务与定位效果：

```bash
python scripts/vlm_target_demo.py --image /实际路径/scene.jpg --instruction "找出左边的香蕉" --no-window
```

每轮成功返回（包括没有找到目标）会保存到 `outputs/vlm_preview/<时间戳>/`：`image.png` 是原始快照，`annotated.png` 是带框图片，`result.json` 包含指令、抓取目标及嵌套 `placement` 放置目标的名称、理由和原图像素坐标。相机模式额外保存同帧深度和内参 `rgbd.npz`，供后续开发使用。输出目录已加入 Git 忽略规则。

`max_image_side` 控制上传尺寸，程序会把结果换算回原图。当前使用 `coordinate_space: normalized_1000`：X、Y 两轴分别归一化到 0～1000，再分别按原图宽、高换算为像素；例如 1280×720 图片中归一化 Y=950 对应像素 Y=684。之前的 Qwen 模型在 `pixel` 提示下仍曾输出归一化数值，因此不能将这些数值直接按像素解析。也可以配置为 `pixel`，但需要实测模型是否遵循。模型必须按所配置的坐标制返回；坐标制不匹配、越界、倒置或零面积的框会被拒绝，并显示模型返回的框、请求尺寸和合法范围。网络错误也不会触发机械臂动作。

第一、二阶段共用“尽力选择一个目标”的 VLM 提示规则：优先直接匹配，其次按用户意图和可能用途匹配，仍无合适物体时选最可能的现有替代目标。只要画面有可定位的可移动物体，就要求返回一个目标；指令模糊、多个物体相似、低置信度或请求物品不存在都不能单独作为返回无目标的理由。没有任何可定位物体（如空桌面、图像不可用）时才返回 `found=false`，不能编造目标框。

例如“我需要杀毒”，桌上只有喷雾瓶和无关物体时优先选喷雾瓶；输出可见名称“喷雾瓶”，理由说明按用途推测、内容物未确认，不能把未知液体直接认定为酒精。无法判断偏好时优先选择轮廓清晰、遮挡少的物体，仍无法区分时选最左侧。这个规则作用于 VLM 抓取目标选择；SAM 分割失败、GraspNet 无合格候选等情况仍会停止本轮抓取位姿输出。

放置目标由 VLM 识别，支持盘子、盒子、碗、托盘、篮子及可见放置表面，不固定使用 `box` 类别。默认同时规划抓取与放置目标，规则为：

- 没有指定放置地点，例如“拿香蕉”：自动选择最适合容纳该物体的可见容器，`selection_mode=auto`，`requested_name` 为空。
- 指定地点且找到，例如“把香蕉放到盘子里”：优先识别盘子，`selection_mode=explicit`，`requested_name=盘子`。
- 指定地点未找到，例如要盘子但只有盒子：选择盒子作为替代，`selection_mode=fallback`，保留 `requested_name=盘子`，实际 `target_name=盒子`；终端说明替代原因，橙色框画在盒子上。
- 明确说“只拿起来，不要放下”：`required=false`、`selection_mode=disabled`，清空放置目标；省略放置要求不会禁用自动选容器。
- VLM 会先严格判断容器当前姿态是否适合接收物体。用于“放入”的容器必须能明确看到入口，入口近似朝上，并允许物体从上方沿近似竖直方向进入；需要从侧面横着塞入、看不到入口、盒子闭合或倒扣、入口朝侧面、严重倾斜、塌陷、入口被挡住，以及入口方向无法可靠判断时均不可用。正常开口朝上、稳定平放的容器可用；它在桌面平面内横放、竖放或斜放都只是方向不同，不会被误判为侧翻。自动或替代选择时，不允许把姿态错误或闭合盒子的外侧板改称为 `relation=on` 的放置表面；只有盘子、托盘等真实朝上的承载面，或者用户明确要求“放在某物上面”时才能使用 `on`。程序会优先寻找另一个姿态正常的可见目的地；如果所有可见目的地都不合格，则返回 `required=false`、`selection_mode=grasp_only`，本轮继续抓取目标的 SAM 和 GraspNet，并分割不可用容器供掩码预览，但不做放置几何规划。不可用容器仍通过 `placement.rejected_container` 返回名称、检测框与原因，画面以红框和框内 `×` 标注；第一阶段保留 `INVALID CONTAINER POSE - GRASP ONLY` 横幅，第二阶段仅显示英文框标签，终端输出中文原因和“仅抓取、不放置”。

同一图片中的抓取框和放置框使用同一个坐标系并分别校验，不允许把同一个物体框当作两个角色。`placement.relation` 记录 `in`（放入）、`on`（放在上面）、`near`（放在旁边）。若没有容器，尽力选择真实可见的合适放置表面；确实没有可定位目的地时保留 `required=true,found=false`，画面显示 `PLACE TARGET NOT CONFIRMED`，`targets_complete=false`。不能把未知或缺失目的地当作完整任务。旧版单目标 JSON 仍可离线解析，新 API 请求必须返回完整 `placement` 结构。

此阶段复用现有 NumPy、OpenCV、PyYAML，英文标签直接由 OpenCV 绘制；相机模式还需要现有相机 SDK。兼容服务请求使用 Python 标准库，不需要额外安装 `openai` SDK。离线测试（不访问服务或硬件）：

```bash
python -m unittest discover -s tests -p 'test_vlm_target.py' -v
```

### 9. `scripts/vlm_grasp_demo.py` — 第二阶段：目标分割＋GraspNet

在终端输入中文指令，VLM 在同一轮识别抓取与放置目标；SAM 分别分割抓取物体和放置容器，GraspNet 仍只对抓取物体生成抓取位姿，容器掩码不会混入抓取候选。程序结合容器掩码与深度估计内部轮廓、三维长短轴和尺寸，再用抓取物体点云搜索一个模拟放置方向。全部步骤共用同一张原始彩色快照，以及同帧、对齐到彩色图的深度和相机内参。该入口仅做识别、分割、抓取和放置预览，不连接机械臂，也不执行抓取或放置。

本机已将参考项目的 `sam_b.pt` 复制到 `models/sam_b.pt`，继续使用现有 Ultralytics。GraspNet 沿用现有 `checkpoint-rs.tar` 和 CUDA 环境。更换机器时需要自行放置这两个权重；程序不会自动下载 SAM 权重。配置如下：

```yaml
sam:
  checkpoint: models/sam_b.pt
  device: cuda:0
  min_mask_area_px: 64
vlm_grasp:
  # Stage 2 中 width：显示 GraspNet 解码后的全部原始候选；best：只显示最终最佳抓取位姿。
  open3d_grasps: best
  context_expand_ratio: 1.35
  depth_tolerance_m: 0.04
  min_target_points: 32
  max_grasp_width_m: 0.09
  placement_preview:
    enabled: true
    min_points: 96
    interior_depth_quantile: 0.50
    min_depth_span_m: 0.018
    plane_distance_threshold_m: 0.008
    floor_normal_tolerance_deg: 25.0
    wall_clearance_m: 0.004
    container_xy_trim_quantile: 0.005
    object_xy_trim_quantile: 0.005
    object_bottom_clearance_m: 0.005
    yaw_step_deg: 1.0
    morph_kernel_px: 7
    max_visual_points: 12000
```

启动：

```bash
cd ~/rebot_grasp
conda activate rebotarm
python scripts/vlm_grasp_demo.py --check-config
python scripts/vlm_grasp_demo.py

# 或启动后自动分析第一张快照，随后继续交互输入。
python scripts/vlm_grasp_demo.py --instruction "把左边那根黄色香蕉放进盒子"
```

总览窗口以相同尺寸的 2×2 区域同时显示：RGB/VLM 标注、抓取物与容器的合并掩码、GraspNet 点云与夹爪、容器放置几何。RGB 区域使用绿色 `PICK TARGET` 表示抓取物、橙色 `PLACE TARGET` 表示放置候选。容器姿态不支持放入（如倒扣、侧翻、封闭、开口阻挡）时，仍显示容器红框和 `INVALID CONTAINER` 标签，并在框内画 `×`；几何尺寸不匹配或几何分析失败也会将容器标红。小尺寸图片上简写为 `PICK`、`PLACE`、`INVALID`。中文目标名称、拒绝原因和放置状态保留在终端与 JSON。每轮完成后可以直接输入新指令；`g`/窗口 `G` 重试新快照，`r`/窗口 `R` 恢复预览，`q`/窗口 `Q`/Esc 退出。`R` 丢弃结果但不会中断正在进行的推理，完成前不能提交下一轮。

不可用容器以 `placement.rejected_container` 单独记录名称、检测框和拒绝原因；有检测框时另行进行 SAM 分割，掩码仅用于显示和保存，不会交给放置几何规划。没有安全替代目标时，仍保持 `selection_mode=grasp_only`，仅预览抓取。存在安全替代目标时，可以同时显示被拒绝的容器与实际放置目标。旧 JSON 未提供不可用容器框时，不会凭空生成框或掩码，UI 会提示重新识别。

默认只打开一个总览窗口。SAM 掩码格把抓取物与容器掩码合并在同一张单通道图中，两者均显示为白色，背景为黑色。放置几何格用橙色显示容器点云、黄色显示完整投影开口轮廓、青色显示可见盒底安全区域、绿色显示通过检查后的模拟香蕉；只有尺寸与安全边距检查通过时才显示绿色。用于“放入”时，规划将稳健物体轮廓置于可见盒底区域内，而不再仅依据包含外壁的 SAM 点云投影开口；无法观察足够盒底时拒绝当前放置位姿。该视图沿盒底法线观察。GraspNet 格显示哪一批夹爪由 `vlm_grasp.open3d_grasps` 决定：第二阶段的 `width` 显示网络刚解码出的全部候选，即碰撞、目标框、掩码/深度和夹爪宽度筛选之前的位姿；`best` 只显示最终一个位姿。左下角会按照“抓取候选中心＋容器三维点云”的联合范围自动裁剪场景并重新适配视野，因此香蕉、夹爪和盒子深度会同时保留，只去掉远处背景与离群深度点；抓取物周围的最小范围由 `vlm_grasp.dashboard.grasp_cloud_focus_radius_m` 调整。四格来自同一轮快照；新快照处理、按 `R` 丢弃、识别失败或目标未找到时会清除旧结果。默认每格为 `640×360`，整个窗口为 `1280×720`，可通过 `vlm_grasp.dashboard.tile_width` 和 `tile_height` 调整。将 `vlm_grasp.dashboard.enabled` 设为 `false` 可恢复独立辅助窗口。

总览支持直接交互：双击任意格可将它放大到整个窗口，再次双击返回四宫格；数字键 `1`～`4` 分别放大左上、右上、左下、右下，`0` 返回四宫格。在左下 GraspNet 点云或右下放置几何区域内，按住鼠标左键拖动可旋转三维视角，按住右键拖动可平移，滚轮可缩放。四宫格和单格放大状态下均可操作。

需要只看最佳位姿或关闭辅助窗口时可以显式指定：

```bash
python scripts/vlm_grasp_demo.py --open3d-grasps final
python scripts/vlm_grasp_demo.py --open3d-grasps best
python scripts/vlm_grasp_demo.py --no-open3d
python scripts/vlm_grasp_demo.py --no-mask-window
```

未传 `--open3d-grasps` 时读取 `default.yaml` 的 `vlm_grasp.open3d_grasps`。第二阶段的 `width` 沿用原配置名，但现在显示 GraspNet 解码后、所有筛选前的原始候选；不执行显示侧 NMS，也不受 `top_k` 截断。`final` 显示通过全部筛选的候选，`best` 只显示按分数选出的最佳位姿。`bbox` 和 `pre-bbox` 用于诊断中间筛选阶段，并使用 `--top-k` 限制去重后的显示数量。原始候选可能与桌面、容器碰撞或不属于抓取目标，仅用于诊断；这些显示选项不改变最终抓取的筛选与选择。已有图片的 `--no-window` 会同时关闭所有二维与三维窗口。

GraspNet 输入保留目标框及附近点云，默认将框扩展为 `context_expand_ratio: 1.35`，提供目标附近的桌面、边缘等几何信息；只采样掩码内点云曾导致部分真实场景中所有候选被碰撞检测剔除。采样前仍要求 SAM 目标掩码内有足够有效深度点，完整深度场景继续用于碰撞检测。最终候选中心投影必须位于目标掩码内，且与对应目标深度的差值不超过 `depth_tolerance_m`，再筛除超过夹爪开口的候选。增加输入上下文不会放宽最终目标归属、碰撞或夹爪宽度检查。

候选不再执行桌面法线或固定朝下方向筛选。GraspNet 输出的旋转矩阵本身已经包含接近方向、夹爪开合方向及绕接近轴的角度，最佳位姿继续按模型分数选择。

SAM 对两个 VLM 框分别调用，掩码都裁在各自框内。抓取目标分割失败时停止本轮；放置目标分割或几何分析失败时保留抓取预览，并在终端和 JSON 中记录放置错误。程序先从两个目标掩码以外的大面积深度点动态拟合同帧桌面法线，再在完整容器点云中寻找与桌面近似平行的平面，要求法线偏差不超过 `floor_normal_tolerance_deg`，防止盒子在图像中横放后 RANSAC 把面积更大的侧壁误认为底面。随后按平面距离扩张并只保留最大的连续盒底区域；只有无法估计桌面法线时才退回原来的深度分层方法。该连续底面负责拟合法线和平面坐标系。完整容器 SAM 点云另投影到盒底平面，并用 `container_xy_trim_quantile` 去除两端少量离群点；该外壁投影仅作诊断，不直接决定放置区域。

放置规划只搜索平面 `X、Y、Yaw`：物体围绕容器法线按 `yaw_step_deg` 搜索方向，并要求稳健平面轮廓保留 `wall_clearance_m` 单侧边距。默认使用 `1°` 步长，避免香蕉等细长目标在临界尺寸下因为 `5°` 采样过粗而漏掉可行角度。`object_xy_trim_quantile` 用于滤除透明瓶等目标的少量深度穿透点。对 `relation=in`，先用可见盒底确定平面与方向，再从高于盒底的容器点云提取盒缘；仅当四侧盒缘都具有足够点数及沿边覆盖，且与可见盒底位置一致时，以内缩 `rim_inset_m` 后的盒缘矩形作为放置区域，从而修正部分遮挡造成的盒底中心偏移。盒缘不完整或冲突时仍按可见盒底保守规划；完整 SAM 投影保留为 JSON 诊断数据，不直接用于放置。Open3D 黄色轮廓是实际采用的规划区域；仅在融合盒缘时，青色轮廓另外标出可见盒底。盒子深度、壁高、物体高度和厚度都不参与 `fits` 判定；`object_bottom_clearance_m` 只影响三维预览的显示高度。当前也尚未检查夹爪与箱壁碰撞、机械臂可达性或执行轨迹；盒缘可信不等于这些安全条件已通过。

终端的 `[候选筛选]` 和 `pipeline.json` 中的 `filter_counts` 记录解码、碰撞后、目标框、掩码内（`mask`）、掩码内有效深度（`valid_depth`）、深度一致（`mask_depth`）和宽度筛选后的累计数量。终端同时显示框内掩码外、深度缺失和深度超差各自剔除的数量；这样可区分候选中心没有落在抓取物上，还是该位置缺少深度或深度误差超过 `depth_tolerance_m`。不改变任何候选的筛选标准；旧结果只含合并的 `mask_depth` 计数，无法反推这三项。放置规划不会改变 GraspNet 的候选筛选和最终抓取位姿。

输出保存在 `outputs/vlm_grasp/<时间戳>/`：

- `image.png`、`annotated.png`：本轮原图，以及包含抓取物/容器检测框、英文标签和不可用标记的标注图；两张图尺寸相同，检测框使用原图坐标。
- `mask.png`：与原图同尺寸的 0/255 目标掩码（找到并成功分割目标时）。
- `placement_mask.png`：VLM 放置框提示得到的原始 SAM 容器掩码。
- `rejected_container_mask.png`：不可用容器的原始 SAM 掩码，仅供预览，不参与放置规划。
- `combined_mask.png`：抓取物、有效放置容器及不可用容器合并后的黑白掩码，存在的目标均为白色。
- `placement_interior_mask.png`：结合深度提取的容器内部区域。
- `rgbd.npz`：同帧毫米深度 `depth_mm` 和内参 `camera_matrix`。
- `result.json`：VLM 选择结果。
- `pipeline.json`：候选数量、最佳相机坐标抓取位姿、完整放置识别结果、容器几何和模拟放置摘要。三维量仍在相机光学坐标系，尚未变换到机械臂基座。
- `placement_geometry.json`：规划开口来源（盒缘/盒底融合或可见盒底回退）、完整 SAM 投影诊断值、连续可见底面尺寸、容器中心、长短轴、法线、拟合残差、平面旋转角度、物体稳健投影尺寸、剩余边距和是否满足 XY/Yaw 放置条件；同时明确记录未检查盒子深度、物体厚度、夹爪碰撞、可达性和实机执行。
- `grasp_candidates.npy`：筛选后的原始 GraspNet 候选数组（每行沿用 GraspNet 的 17 维格式）。
- `grasp_cloud.png`：本轮点云与夹爪的三维截图，仅在 Open3D 窗口成功显示时保存；实际截图模式记录在 `pipeline.json` 的 `open3d_grasps` 字段。
- `placement_cloud.png`：容器三维几何和模拟物体轮廓截图，仅在放置 Open3D 窗口成功显示时保存。
- `dashboard.png`：与程序窗口一致的 2×2 四画面总览图。

已有 RGB-D 快照可离线回放相机数据；该命令仍会请求配置的远程 VLM。图像、深度必须来自同一轮，程序会检查尺寸，但无法验证手动指定文件的拍摄时间：

```bash
python scripts/vlm_grasp_demo.py \
  --image outputs/vlm_preview/某次结果/image.png \
  --rgbd outputs/vlm_preview/某次结果/rgbd.npz \
  --instruction "我想玩积木" --no-window
```

现有 `grasp.py`、`grasp_set.py` 继续使用原来的 YOLO 选目标流程。后续接机械臂时，可以直接调用 `VLMGraspPipeline.process()` 得到 `grasp_result.best`，复用现有位姿变换和放置流程。底层 `utils.graspnet_utils.infer_frame()` 也支持外部 `target_bbox_xyxy`、`target_mask`，两者均使用原图坐标，不能与 YOLO 目标选择同时传入；单独传框只限制框范围，传掩码才会使用掩码及深度筛选。

### VLM 框中心固定姿态放置对照组

`scripts/vlm_grasp_box_center_robot.py` 是独立于容器几何规划的实机对照入口。抓取仍使用 VLM＋SAM＋GraspNet，以及桌面法线、抓取方向和 IK 检查；放置只使用 VLM 确认的、可从上方进入的容器框。取框中心像素附近 11×11 深度的中位数反投影到相机坐标，再通过手眼标定转换到基座坐标，仅取其 XY。释放位姿沿用 `set.py`：基座 Z 固定为 `0.08 m`，RPY 固定为 `(0, 1.2, 0)`；pre-place 与释放点 XY/RPY 相同、Z 高出 `vlm_robot.place_approach_offset_m`。抓取、抬高、接近、释放和返回 ready 的运动安全流程复用 `vlm_grasp_robot.py`，但不计算盒子姿态、开口尺寸、物体 fit 或盒壁碰撞，也不进行抬高后的容器几何复查。VLM 判为不可用的容器、中心附近无有效深度或放置 IK 不通过时，不执行抓取或放置。

```bash
python scripts/vlm_grasp_box_center_robot.py --instruction "把香蕉放进盒子" --dry-run
python scripts/vlm_grasp_box_center_robot.py --instruction "把香蕉放进盒子"
```

默认结果写入 `outputs/vlm_box_center/`，其中 `box_center_baseline.json` 记录中心像素、深度样本数、相机坐标、固定基座释放位姿和 IK 结果，便于与几何规划组比较。左下 GraspNet 点云格用 VLM 容器框从完整 RGB-D 点云选取盒子点以扩展视野；即使对照组不运行容器 SAM，仍能同时看到抓取物和盒子。**该对照组只检查 IK，不验证固定高度与实际箱底/箱壁的碰撞；实机运行前先检查 `--dry-run` 的目标位姿和现场安全条件。**

### 单指令连续抓取放置

`scripts/vlm_grasp_sequence_robot.py` 是基于几何放置实机流程的独立入口，保留原单次脚本。一次输入“把桌面盒外的香蕉全部放进盒子，先从最外围开始”，每轮先拍新的 RGB-D，VLM 按当前画面重新清点盒外目标并锁定本轮抓取框，再执行原有的 SAM、GraspNet、桌面法线、抓取 IK、容器分割／几何和放置 IK 检查。VLM 不假设上轮抓取成功，也允许新增目标；盒外数量不变或增加不会被当作任务失败，只要仍有目标且安全检查通过，就继续按原指令抓取。首次通过预览检查后需按一次 `G`/空格确认；以后只有确认松爪并返回 ready 才自动拍新图进入下一轮。抬高后必要时仍会重拍图复查容器。连续两张新图清点为空才判定完成，且确认无持物、放置复核已结束后，自动返回配置的 ready 位并退出程序；归位失败会以非零状态退出并提示人工检查。容器不确定、掩码／位姿／IK 检查失败或达到步数上限且仍有目标时暂停，不会强行继续。视觉网络失败时可按 `g` 重拍检查、`r` 放弃本轮、`q` 退出；手动重试检查通过后还需再按一次 `g` 执行。抬高复查核对抓取前后的容器位置，名称描述变化不再单独阻止放置；改用其他容器仍会暂停。`--dry-run` 不发送运动命令，也不会模拟完成多轮。按 `Q`/Esc 退出；故障时先人工确认夹爪和机械臂状态。

连续清点要求 VLM 根据盒沿、盒壁、桌面支撑和遮挡关系区分盒外、盒内与不确定目标；二维检测框重叠不再用于判定香蕉是否在盒内。VLM 判为盒外的目标继续进入抓取安全检查；判为盒内的目标不抓取；无法确认位置时暂停并允许重新拍照。

RS 夹爪只使用固定的闭合／张开位置指令，不再把 GraspNet 宽度换算为电机角度。抓取前读取夹爪反馈：若已张开过半，先位置控制归零、确认到位，再按原 Demo 方式张开；若驱动仍处于夹持状态则拒绝重新开合。放置松爪保持原流程，不先闭合。RS 电机命令硬上限为 `4.71 rad`，默认 `angle_open=4.71 rad` 沿用 Demo 的 `0.98` 系数，张开目标约 `4.616 rad`；对应配置为 `robot.gripper.rs.max_open_angle_rad`。张开超时会撤销外推命令并停止抓取或放置。**硬件已经损坏时，软件限幅不能代替维修或重新标定；确认机械结构、编码器零点及真实安全开度前不要再次上电执行抓放。**实机运行时必须现场监护夹爪、物体及运动路径，不可无人值守。

退出并回 Home 时，断开控制前会读取夹爪位置反馈：已在零位则不重复下发，未归零则位置控制回到 `0 rad` 并等待到位；超时会停止夹爪驱动并报警，机械臂仍尝试 Home。VLM 流程若仍持物、放置待复核或处于故障状态，不自动闭合夹爪，避免误伤物体；需人工处理。

VLM 实机抓取现将相对桌面法线的最大接近倾角设为 `40°`；过于侧向的候选直接拒绝，更垂直的候选在排序中优先。对香蕉，还使用 SAM 掩码的长轴位置和中心到掩码边界的近似物理距离：排除两端各 `15%`、距侧边不足 `6 mm` 的候选。网络分数达到本轮安全候选最高分 `55%` 的中段（长轴 `30%~70%`）候选优先，其余按网络分数、中段位置、内部余量和接近倾角排序；中段 IK 不可达时仍可尝试非端部候选。`40°` 来自近期保存的离线候选对照：`35°` 会误删一些仍接近中段的候选，而旧 `50°` 允许过于侧向的位姿。运行时的「香蕉中段审计」及 `pipeline.json` 中的 `filter_counts.banana_middle_*` 会分别统计解码、碰撞、目标框、掩码/深度、宽度、接近角和 ready 姿态各阶段剩余的中段候选；早期阶段的中段计数只反映投影位置与掩码余量，不代表已通过碰撞、深度或 IK。若没有符合条件且 IK 可达的候选，就暂停本轮，不退回抓头尾或贴边姿态。参数见 `vlm_robot.max_approach_tilt_deg`、`vlm_robot.banana_end_fraction` 和 `vlm_robot.banana_min_mask_clearance_m`。这是图像空间的保守筛选，不验证实际夹爪两侧接触点或整条运动轨迹；必须先在 `--dry-run` 和预览中检查，再实机监护验证。

```bash
python scripts/vlm_grasp_sequence_robot.py --instruction "把桌面盒外的香蕉全部放进盒子，先从最外围开始" --dry-run
python scripts/vlm_grasp_sequence_robot.py --instruction "把桌面盒外的香蕉全部放进盒子，先从最外围开始" --max-steps 5
```

默认结果按时间戳写入 `outputs/vlm_sequence_robot/`。目标顺序由 VLM 根据完整指令和每轮新图理解，例如“先从最外围开始”；程序按其返回的列表顺序选择首个盒外目标，不再套用固定的图像距离公式。遮挡或空间关系复杂时应先人工确认首轮目标。实机前先运行 `--check-config` 和 `--dry-run`，再清空机械臂运动路径。

腕部相机录像在 `config/default.yaml` 的 `vlm_robot.record_video` 中开关，当前默认 `true`；关闭时保持原有直接采集方式。三个 VLM 实机入口（单次、连续、盒中心对照）将原始彩色画面写入各自 `--output-dir/recordings/wrist_rgb_*.mp4`，录制线程在机械臂运动阻塞主循环时仍持续采集。每段视频旁的同名 `.json` 记录写入帧数、无彩色/深度帧数、采集错误、录像写入错误，以及实际采集时间跨度与按配置帧率计算的视频时长；两种时长差异过大表示视频播放可能加速，不等同于相机断连。视频不包含深度图或四宫格标注；正常退出时自动关闭文件。`--dry-run` 也会录制。

连续任务在放置后先检查夹爪反馈是否沿张开方向移动至少 `release_min_open_motion_rad`，不要求达到旧版张爪命令目标；未确认张开时不退让、不返回 ready、不启动下一轮。返回 ready 后重新获取 RGB-D，要求新图的 VLM 清点明确没有夹爪持物，才计入一次放置并规划下一个目标；若视觉失败或夹爪仍持物，保留复核状态，只允许按 `g` 重拍或退出，不能用 `r` 或新指令绕过。盒外数量未减少并不代表任务失败，仍按原指令规划。编码器运动和 VLM 观察都不是物体确实落入盒内的独立物理验证；VLM 漏检、遮挡、夹爪粘连或未建模的轨迹碰撞仍需现场监护，不能无人值守运行。连续任务受 `--max-steps` 已确认放置次数上限约束。

若抓取反馈未确认物体但机械臂已返回 ready，程序暂停并明确提示：确认夹爪确实无物后按 `g` 重新拍 RGB-D、重新规划并再次确认执行，或按 `r` 放弃本轮；不会沿用上次的抓取位姿，也不会自动执行放置。若机械臂未确认返回 ready，则进入故障状态，禁止重试。

全部离线回归测试：

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
```

---

## default.yaml 参数说明

#### 1. 相机与标定配置 (`camera` & `calibration`)

| 参数项 | 类型 / 可选值 | 含义与详细说明 |
| :--- | :--- | :--- |
| `camera.type` | `realsense_d435i`<br/>`realsense_d405`<br/>`orbbec_gemini2` | **相机类型**：指定当前系统连接的相机硬件。 |
| `camera.serial` | `string` / `null` | **设备序列号**：指定具体的设备 SN 号。填写 `null` 表示默认使用系统检测到的第一台可用设备。 |
| `calibration.aruco.marker_length_m` | `float` | **ArUco 边长**：手眼标定所使用的 ArUco 标定码实际物理边长，单位为**米 (m)**。 |
| `calibration.hand_eye_compensation_m` | `array` | **手眼标定平移补偿**：手眼标定完成后，在**机器人基坐标系**下执行的 XYZ 手动平移补偿（格式为 `[X, Y, Z]`），单位为**米 (m)**。若三项全为 `0.0`，则补偿矩阵为单位矩阵。 |

---

#### 2. 目标检测配置 (`detection`)

| 参数项 | 类型 | 含义与详细说明 |
| :--- | :--- | :--- |
| `detection.conf_threshold` | `float` | **YOLO 检测置信度阈值**：得分低于该值的检测框将被过滤。 |
| `detection.iou_threshold` | `float` | **YOLO NMS IoU 阈值**：非极大值抑制（NMS）中用于过滤重叠框的交并比（IoU）阈值。 |

---

#### 3. 机器人与夹爪配置 (`robot`)

| 参数项 | 类型 / 可选值 | 含义与详细说明 |
| :--- | :--- | :--- |
| `robot.repo_root` | `string` / `null` | **仓库根目录**：`reBotArm_control_py` 仓库的路径。为 `null` 时默认使用内部相对路径 `sdk/reBotArm_control_py`。 |
| `robot.ready_pose` | `array` | **预备位**：系统机械臂启动后优先到达的预备位置。每次抓取任务结束后，机械臂也会自动返回至该位置。 |
| `robot.gripper.dm`<br/>`robot.gripper.rs` | 结构体对象 | **夹爪硬件参数**：系统会根据 SDK 当前的实际硬件配置，自动选择并应用对应的这两组参数之一。 |

#### 夹爪内部核心参数说明

对于 `robot.gripper.dm` 或 `robot.gripper.rs` 内部的子参数，定义规则如下：

* **`angle_open`**、**`close_torque`**、**`default_force`**：分别对应电机张开角度（弧度 rad）、闭合阶段命令力矩和持续夹持阶段命令力矩，均需填写**正数数值**。当前 RS 的持续夹持力矩由 `0.30` 小幅调至 `0.40` Nm，闭合命令仍为 `1.5` Nm；电机力矩不等于实际爪口夹持力。`angle_open` 的硬件风险见上文提示。
* **`counterclockwise`**：布尔值。指示闭合时采用的电机转动方向（是否为逆时针）。代码会据此逻辑自动推导出张开角度和闭合力矩的符号。
* **`tau_max`**：力矩上限。

:::tip
注：其余更高级的夹爪控制行为参数，可在 `drivers/robot/grasp_driver.py` 文件中进行查阅与定义。
:::

---

#### 4. 抓取管线与 GraspNet 配置 (`grasp_pipeline` & `graspnet`)

| 参数项 | 类型 | 含义与详细说明 |
| :--- | :--- | :--- |
| `grasp_pipeline.infer_every_live` | `int` | **推理帧间隔**：在实时视频预览时，每 N 帧运行一次目标检测，用以有效减轻 CPU/GPU 的实时计算压力。 |
| `grasp_pipeline.grasp.depth_quantile` | `float` | **深度分位数**：短轴抓取管线所使用的深度计算分位数。该数值越大，计算出的抓取点通常会越深。 |
| `grasp_pipeline.grasp.pregrasp_offset_m` | `float` | **预抓取位偏移量**：相对于最终的目标抓取位置，沿末端进给方向往回退的距离，单位为**米 (m)**。 |
| `grasp_pipeline.grasp.insertion_depth_m` | `float` | **插入深度**：普通 YOLO 抓取（`scripts/main.py`、`scripts/set.py`）及 GraspNet 抓取沿末端进给方向额外推入的距离，单位为**米 (m)**。该方向随夹爪姿态变化，不一定沿基座 Z 轴。 |
| `grasp_pipeline.grasp.min_base_z_m` | `float` | **最低抓取高度限制**：在**机械臂基坐标系**下允许的最低抓取 Z 轴高度，单位为**米 (m)**（作为底层防碰撞的安全边界）。 |
| `graspnet` | 结构体配置 | **GraspNet 运行参数**：该配置项下的所有子参数供 `scripts/graspnet_camera_demo.py` 和 `scripts/grasp.py` 脚本运行时加载使用。 |

### 模型选择库

YOLO 模型会从 `rebot_grasp/models/` 目录加载；如果模型文件不存在，Ultralytics 通常会尝试自动下载。

常用模型：

| 模型 | 说明 |
| --- | --- |
| `yoloe-26l-seg.pt` | 开放词汇 + 分割，当前默认 |
| `yoloe-26s-seg.pt` | 更轻量，速度更快 |
| `yolov8n-seg.pt` | 封闭类别分割，小模型 |
| `yolov8s-seg.pt` | 封闭类别分割，精度更高 |

当模型名包含 `world` / `yoloe`，并且 `yolo.use_world=true` 时，程序会调用 `model.set_classes(custom_classes)`，将 `yolo.custom_classes` 注入为开放词汇类别。普通 `yolov8*-seg.pt` 模型会忽略这组开放词汇类别。

---

## ❓ FAQ

### 1. `ModuleNotFoundError: No module named 'motorbridge'`

这通常表示当前 Python 环境还没有安装机械臂 SDK 依赖。请确认已激活项目环境，并重新同步环境与安装机械臂 SDK：

```bash
conda activate rebotarm
conda env update -n rebotarm -f environment.yml
cd sdk/reBotArm_control_py && pip install -e .
```

### 2. 按 `G` 后不执行抓取

常见原因包括：

- `hand_eye.npz` 不存在
- 手眼标定模式不是 `eye_in_hand`
- 当前目标位姿 IK 不可达

建议先使用 dry-run 模式验证感知结果和目标位姿：

```bash
python scripts/main.py --dry-run
```

### 3. 抓取点深度不稳定

可以优先检查和调整：

- `grasp_pipeline.grasp.depth_quantile`
- 相机与目标工作区域的安装高度
- 目标表面的反光情况

### 4. GraspNet 报 `pointnet2_utils` 无法从 `pointnet2` 导入

这通常是 `sdk/graspnet-baseline/pointnet2` 本地 CUDA 扩展没有在当前 conda 环境中正确编译安装，或 Python 路径解析到了错误的 `pointnet2` 包。建议确认已激活项目环境，并在同一个环境中重新编译安装 `pointnet2` 与 `knn`：

```bash
conda activate rebotarm
cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

验证：

```bash
python -c "from pointnet2 import pointnet2_utils; print('Submodule import works')"
```

### 5. 当前显卡运行 GraspNet 时出现 CUDA 架构不兼容

如果出现 `no kernel image is available for execution on the device` 或 PyTorch 提示当前 GPU 的 CUDA capability 不受支持，通常说明当前 PyTorch wheel 不包含该显卡架构对应的 CUDA kernel。建议安装支持当前 CUDA/显卡架构的 PyTorch 版本，然后重新编译 GraspNet 的本地 CUDA 扩展。

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

cd sdk/graspnet-baseline/pointnet2
pip install . --no-build-isolation

cd ../knn
pip install . --no-build-isolation
```

如需手动指定编译架构，可在重新编译前设置 `TORCH_CUDA_ARCH_LIST`，具体取值请按当前显卡架构和 PyTorch/CUDA 版本确认。

### 6. GraspNet 推理时报 `RuntimeError: CPU not supported`

`pointnet2` 中的采样算子只支持 CUDA tensor。请确认 CUDA 可用、GraspNet 网络和输入点云都在 GPU 上，并且 `pointnet2` / `knn` 是在当前环境和当前 PyTorch 版本下编译的。

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

如果输出为 `False`，需要先修复 CUDA / PyTorch 安装；如果输出为 `True` 但仍报错，建议重新编译 `pointnet2` 和 `knn`。

---

## 源码发布范围与许可

源码仓库不包含运行结果：`outputs/`、`dataset/` 下的截图与录像、模型权重、下载的 SDK、本地密钥、缓存和 `*.npz` 标定数据均不分发。模型请按上文自行下载，标定请按自己的设备重新采集。默认 YAML 保留开发配置示例，需按实际设备核对。

goatwzz 的原创贡献采用 [MIT License](LICENSE)，上游和第三方组件继续遵循各自授权，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## 📄 参考资料

- [reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py) — 机械臂控制库
- [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) — reBot 机械臂开源项目
- [Orbbec Gemini 2 产品页](https://www.orbbec.com.cn/index/Product/info.html?cate=38&id=51)
- [Orbbec SDK v2](https://github.com/orbbec/OrbbecSDK_v2)
- [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- [RealSense SDK](https://github.com/realsenseai/librealsense)
- [graspnet/graspnet-baseline](https://github.com/graspnet/graspnet-baseline)
- [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics)

---

## ☎ 联系我们

- **技术支持**：[提交 Issue](https://github.com/goatwzz/rebot_grasp/issues)

---

<p align="center">
  <strong>🌟 如果本项目对你有帮助，欢迎点个 Star！</strong>
</p>
