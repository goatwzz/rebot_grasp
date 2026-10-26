# Optional hook for rebotarm/etc/conda/activate.d/zz_rebotarm_can.sh.
# Set REBOT_GRASP_ROOT to the checkout path when copying this hook into Conda.
# Keep activation successful even if hardware is unplugged or sudo is unavailable.
rebot_can_project_root="${REBOT_GRASP_ROOT:-}"
if [ -z "$rebot_can_project_root" ]; then
    rebot_can_project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
if [ -f "$rebot_can_project_root/scripts/setup_can.sh" ]; then
    if ! bash "$rebot_can_project_root/scripts/setup_can.sh"; then
        printf '[CAN] 自动初始化未完成；Conda 环境已激活。抓取前请手动检查 can0。\n' >&2
    fi
else
    printf '[CAN] 找不到项目的 scripts/setup_can.sh；请设置 REBOT_GRASP_ROOT 为仓库路径，跳过 CAN 初始化。\n' >&2
fi
unset rebot_can_project_root
