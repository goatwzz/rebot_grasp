#!/usr/bin/env bash
# Initialize the physical CAN interface before starting reBot control.
# Run as a subprocess (also used by the rebotarm Conda activation hook).

rebot_setup_can() {
    local can_iface="can0"
    local can_bitrate="1000000"
    local can_ip can_status can_first_line can_flags can_sudo
    local -a can_command

    can_ip="$(command -v ip)" || {
        printf '[CAN] 找不到 ip 命令，请安装 iproute2。\n' >&2
        return 1
    }
    if ! can_status="$("$can_ip" -details link show "$can_iface" 2>&1)"; then
        printf '[CAN] 无法读取 %s；请检查 USB-CAN 连接和接口名称。\n%s\n' "$can_iface" "$can_status" >&2
        return 1
    fi

    can_first_line="${can_status%%$'\n'*}"
    can_flags="${can_first_line#*<}"
    can_flags="${can_flags%%>*}"
    if [[ ",$can_flags," == *,UP,* ]] &&
       [[ "$can_status" =~ bitrate[[:space:]]+1000000([[:space:]]|$) ]]; then
        printf '[CAN] can0 已启动，bitrate=1000000，沿用当前配置。\n'
        printf '%s\n' "$can_status"
        return 0
    fi

    can_command=("$can_ip")
    if [[ "$EUID" -ne 0 ]]; then
        can_sudo="$(command -v sudo)" || {
            printf '[CAN] 配置接口需要 sudo，但未找到 sudo 命令。\n' >&2
            return 1
        }
        if ! "$can_sudo" -n -v 2>/dev/null; then
            if [[ ! -t 0 ]]; then
                printf '[CAN] 需要 sudo 授权；请在仓库根目录的交互终端运行 bash scripts/setup_can.sh。\n' >&2
                return 1
            fi
            printf '[CAN] 初始化 can0 需要 sudo 权限，请按提示输入系统密码。\n'
            "$can_sudo" -v || return 1
        fi
        can_command=("$can_sudo" "$can_ip")
    fi

    printf '[CAN] 配置 %s，bitrate=%s...\n' "$can_iface" "$can_bitrate"
    "${can_command[@]}" link set "$can_iface" down || return 1
    "${can_command[@]}" link set "$can_iface" type can bitrate "$can_bitrate" || return 1
    "${can_command[@]}" link set "$can_iface" up || return 1
    "$can_ip" -details link show "$can_iface"
}

rebot_setup_can
