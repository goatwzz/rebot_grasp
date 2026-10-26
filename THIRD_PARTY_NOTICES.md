# 上游来源与第三方授权说明

## 基础代码来源

本项目基于以下仓库扩展，保留对原作者和维护者的致谢：

- https://github.com/Seeed-Projects/reBot-DevArm-Grasp
- 原始开发仓库：https://github.com/EclipseaHime017/reBot-DevArm-Grasp
- 本地开发起点：`29bdff8`（`Merge pull request #4 from TheMoonAstronaut/main`）。

`calibration/`、`drivers/`、传统 YOLO / GraspNet 脚本及部分 `utils/` 来自上述项目，部分文件在本项目中有修改。`docs/UPSTREAM_GUIDE_EN.md` 保留原英文教程内容，教程中的型号、服务、链接和行为可能与本项目当前版本不同。新增 VLM、SAM、容器几何、连续任务和相关测试以本仓库当前说明为准。

上游 README 标注 MIT，但截至 2026-10-26，所检查的上游公开仓库及本地基线没有 `LICENSE` 正文，GitHub API 也未识别出许可证。此事实不应被解释为已取得对上游代码重新授权的权利。根目录 `LICENSE` 仅覆盖 goatwzz 的原创贡献；使用、再分发或商用上游部分前，请向相应权利人核实授权范围。

## 外部依赖

以下组件及其模型权重不包含在本次源码发布包中，需自行按官方说明获取，并遵守各自许可证：

| 组件 | 官方来源 | 授权说明 |
| --- | --- | --- |
| reBot 机械臂 SDK | https://github.com/vectorBH6/reBotArm_control_py | 以该 SDK 发布的许可证为准 |
| GraspNet baseline | https://github.com/graspnet/graspnet-baseline | 当前本地 SDK 许可证正文限制为非商业内部研究用途，且包含第三方组件条款；不属于整套 MIT 许可 |
| GraspNet API | https://github.com/graspnet/graspnetAPI | 独立项目，以其许可证和数据 / 模型条款为准 |
| Ultralytics YOLO / SAM 接口 | https://github.com/ultralytics/ultralytics | AGPL-3.0 / 商业许可，具体取决于所用版本和授权方式 |
| SAM 模型 | https://github.com/facebookresearch/segment-anything | 模型与分发封装的条款应分别核对 |
| Orbbec SDK | https://github.com/orbbec/pyorbbecsdk | 以 SDK 和二进制分发条款为准 |
| RealSense SDK | https://github.com/realsenseai/librealsense | 以相应版本的 SDK 许可证为准 |
| PyTorch、OpenCV、NumPy 等 | 见 `environment.yml` 与各项目官网 | 各自许可证继续适用 |

本项目的 MIT 许可不会覆盖第三方组件，也不会更改模型服务商的服务条款。VLM 请求通过用户自行配置的远程服务发送指令和图像；API 密钥由用户自行配置，不随源码分发。
