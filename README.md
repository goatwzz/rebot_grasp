# rebot_grasp：自然语言驱动的机械臂视觉抓取与放置

基于 reBot Arm B601、RGB-D 相机与 **VLM + SAM + GraspNet** 的桌面操作项目。输入中文指令，系统识别抓取物与放置容器，估计三维抓取姿态，检查放置几何与机械臂可达性，再执行抓取和放置。

**[观看 B 站实机演示](https://www.bilibili.com/video/BV1iZH36UEpJ/)** · [详细安装与参数说明](README_zh.md) · [上游英文教程](docs/UPSTREAM_GUIDE_EN.md)

本项目由 [goatwzz](https://github.com/goatwzz) 在 [Seeed-Projects/reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp) 基础上扩展，保留原有 YOLO / OBB 抓取流程，新增自然语言目标选择、目标分割、容器几何规划、连续抓取放置与实机复查功能。上游及依赖的授权情况见 [第三方说明](THIRD_PARTY_NOTICES.md)。

## 演示视频

- Bilibili：[机械臂抓取与放置展示 · BV1iZH36UEpJ](https://www.bilibili.com/video/BV1iZH36UEpJ/)
- 示例指令：`把左边那根黄色香蕉放进盒子`。
- 连续任务示例：`把桌面盒外的香蕉全部放进盒子，先从最外围开始`。

## 功能

| 功能 | 说明 |
| --- | --- |
| 自然语言目标选择 | VLM 根据当前 RGB 快照与中文指令返回抓取物、放置目标和二维框 |
| 目标分割与抓取估计 | SAM 获取目标掩码，GraspNet 生成三维候选，经目标、深度、碰撞和夹爪宽度筛选 |
| 容器几何规划 | 使用 RGB-D 估计可见盒底、开口方向与安全放置区域，搜索可容纳物体的 XY / Yaw 方案 |
| 实机抓取放置 | 手眼标定坐标变换、桌面法线约束、IK 检查、抓取后抬高复查容器、释放与返回 ready |
| 连续任务 | 每轮重新拍照清点目标；连续两张新图确认无目标后结束，支持步数上限 |
| 对照流程 | 提供 VLM 框中心 + 固定高度放置入口，便于与几何规划流程比较 |
| 传统视觉流程 | 保留 YOLO / OBB、GraspNet 相机预览和机械臂抓取脚本 |
| 可视化与录像 | 四格总览显示 RGB、掩码、抓取点云和放置几何，可录制腕部相机视频 |

```text
中文指令 + 同帧 RGB-D
        ↓
VLM：抓取目标 / 放置目标 / 二维定位
        ↓
SAM：目标与容器分割
        ↓
GraspNet：三维抓取候选 → 目标与几何筛选
        ↓
容器放置规划 + 手眼坐标变换 + IK 检查
        ↓
预览确认 → 抓取 → 抬高复查 → 放置 → 返回 ready
```

## 硬件与环境

- reBot Arm B601（DM / RS，按实际型号配置控制 SDK）。
- Intel RealSense D435i / D405 或 Orbbec Gemini 2 深度相机。
- Ubuntu 22.04+、Python 3.10、Conda。
- GraspNet 推理需要 NVIDIA GPU、CUDA 版 PyTorch 和匹配的 CUDA 编译器。
- VLM 流程需要支持图像输入的兼容 Chat Completions 服务；指令和图像会发送到你配置的服务地址。

## 安装

```bash
git clone https://github.com/goatwzz/rebot_grasp.git
cd rebot_grasp
conda env create -f environment.yml
conda activate rebotarm

# 机械臂 SDK
git clone https://github.com/vectorBH6/reBotArm_control_py.git sdk/reBotArm_control_py
pip install -e sdk/reBotArm_control_py

# 根据相机型号选择其一
pip install pyrealsense2
# pip install pyorbbecsdk2
```

GraspNet 的 CUDA 算子、API 与 PyTorch 安装步骤见 [详细教程](README_zh.md#step-5-配置-graspnet可选)。VLM + SAM + GraspNet 流程必须完成此步骤；仅运行 VLM 目标定位预览时不需要这些模型。

模型权重不随源码分发，需自行下载：

| 权重 | 默认位置 | 来源 |
| --- | --- | --- |
| GraspNet RealSense checkpoint | `sdk/graspnet-baseline/checkpoints/checkpoint-rs.tar` | [GraspNet 官方项目](https://github.com/graspnet/graspnet-baseline#pretrained-models) |
| SAM base | `models/sam_b.pt` | [Ultralytics SAM 文档](https://docs.ultralytics.com/models/sam/) |
| YOLO / YOLOE | `models/` 下，文件名与 `yolo.model_name` 一致 | [Ultralytics](https://docs.ultralytics.com/models/) |
| MobileCLIP（开放词汇流程按需） | `mobileclip2_b.ts` | 按所用 Ultralytics 模型的官方说明下载 |

`models/` 与 `sdk/` 可以在安装或下载时创建。SAM 加载器支持通过 Ultralytics 下载所配置的权重；离线运行前请准备完整模型文件。

## 配置

编辑 `config/default.yaml`：选择相机类型，配置机械臂 SDK、模型路径、设备和抓取 / 放置参数。在 `sdk/reBotArm_control_py/config/rebotarm.yaml` 中选择实际的 DM / RS 硬件配置。

VLM 地址和模型可在 YAML 的 `vlm` 段配置，或使用环境变量覆盖。默认 YAML 保留开发时的服务示例，需根据自己的服务账号调整：

```bash
export VLM_BASE_URL="https://your-provider.example/v1"
export VLM_MODEL="your-vision-model-id"
read -rsp "VLM API Key: " VLM_API_KEY
export VLM_API_KEY
printf '\n'
```

也可以将密钥单独写入 `config/vlm_api_key.local`，并设置 `chmod 600 config/vlm_api_key.local`。环境变量优先；本地密钥文件已被 Git 忽略。

实机运行前完成自己的手眼标定：

```bash
python scripts/collect_handeye_eih.py
# 或手动采样
python scripts/collect_handeye_eih.py --manual
```

生成的内参和手眼标定保存在 `config/calibration/<camera_type>/`，不随源码发布。不同相机和安装位置需分别标定；当前 YAML 中的手眼补偿、ready 位和运动参数是开发配置，需按自己的设备检查。

## 运行

以下命令均在仓库根目录、激活 `rebotarm` 环境后执行。

```bash
# 1. VLM 目标定位：不连接机械臂
python scripts/vlm_target_demo.py --check-config
python scripts/vlm_target_demo.py --instruction "把左边那根黄色香蕉放进盒子"

# 2. 分割、抓取与放置几何预览：不连接机械臂
python scripts/vlm_grasp_demo.py --check-config
python scripts/vlm_grasp_demo.py --instruction "把香蕉放进盒子"

# 3. 实机流程 dry-run：连接机械臂并检查 FK/IK，不发送运动或夹爪命令
python scripts/vlm_grasp_robot.py --instruction "把香蕉放进盒子" --dry-run

# 4. 单次实机抓取放置
python scripts/vlm_grasp_robot.py --instruction "把香蕉放进盒子"

# 5. 单指令连续任务
python scripts/vlm_grasp_sequence_robot.py \
  --instruction "把桌面盒外的香蕉全部放进盒子，先从最外围开始" \
  --max-steps 5

# 6. 框中心固定高度放置对照组
python scripts/vlm_grasp_box_center_robot.py --instruction "把香蕉放进盒子" --dry-run
```

预览中 `G` / 空格触发或确认当前轮，`R` 丢弃本轮恢复预览，`Q` / Esc 退出。连续任务首轮确认后，在完成释放并返回 ready 后自动进入下一轮。具体交互、暂停和复查行为见 [详细说明](README_zh.md)。

实机操作会移动机械臂；先检查配置、标定和 dry-run 输出，清空运动路径并准备好急停。固定高度对照组不进行容器几何和箱壁碰撞检查，使用前需核对实际容器高度。

## 代码结构

```text
rebot_grasp/
├── calibration/                 # ArUco 与手眼标定
├── config/default.yaml         # 主配置
├── drivers/                    # 相机与机械臂驱动
├── scripts/
│   ├── vlm_target_demo.py      # VLM 定位预览
│   ├── vlm_grasp_demo.py       # 分割、抓取与放置几何预览
│   ├── vlm_grasp_robot.py      # 单次实机抓取放置
│   ├── vlm_grasp_sequence_robot.py  # 连续任务
│   ├── vlm_grasp_box_center_robot.py # 固定姿态放置对照
│   ├── grasp.py / grasp_set.py # 传统 GraspNet 流程
│   └── collect_handeye_eih.py  # 标定采集
├── utils/                      # VLM、SAM、抓取筛选、放置几何与录像
├── tests/                      # 离线单元测试
├── docs/                       # 上游英文教程
├── environment.yml
├── README_zh.md                # 详细中文教程
├── LICENSE                     # 本项目原创贡献的 MIT 许可
└── THIRD_PARTY_NOTICES.md       # 上游来源与依赖授权说明
```

`outputs/`、`dataset/`、截图、录像、标定结果、本地密钥、模型权重、下载的 SDK 和缓存不上传。程序会在运行时自动生成自己的结果目录；代码中使用合成数据和 mock 的测试随仓库保留。

## 离线测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 VLM 返回格式与坐标校验、目标掩码、抓取筛选、容器几何、实机流程状态、连续任务和录像等。完整测试需要先按安装教程准备 GraspNet SDK 及其运行依赖；不访问相机、机械臂或模型服务。离线测试不代表已在你的设备上完成实机验证。

## 来源与许可

- 基础项目：[Seeed-Projects/reBot-DevArm-Grasp](https://github.com/Seeed-Projects/reBot-DevArm-Grasp)，其原始开发仓库为 [EclipseaHime017/reBot-DevArm-Grasp](https://github.com/EclipseaHime017/reBot-DevArm-Grasp)。
- 控制库：[reBotArm_control_py](https://github.com/vectorBH6/reBotArm_control_py)。
- 三维抓取：[GraspNet baseline](https://github.com/graspnet/graspnet-baseline) / [GraspNet API](https://github.com/graspnet/graspnetAPI)。
- 分割与检测：[Ultralytics](https://github.com/ultralytics/ultralytics)。

本项目由 goatwzz 新增的原创代码和文档采用 [MIT License](LICENSE)。上游代码的原有权利归其作者所有；上游 README 标注 MIT，但当前核查未发现许可证正文，不能据此承诺整个代码栈可任意商用。GraspNet、Ultralytics、SDK 与模型权重遵循各自条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
