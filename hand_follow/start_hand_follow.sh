#!/usr/bin/env bash
# 统一启动：脸跟踪常开 + 手势1撒娇 + 手势2~4动作库（五指跟手已禁用）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIRD_WS="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -f /opt/ros/noetic/setup.bash ]; then
  # shellcheck source=/dev/null
  source /opt/ros/noetic/setup.bash
fi
if [ -f "${HOME}/sim2real/devel/setup.bash" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/sim2real/devel/setup.bash"
elif [ -f "${HOME}/sim2real/install/setup.bash" ]; then
  # shellcheck source=/dev/null
  source "${HOME}/sim2real/install/setup.bash"
fi

export DISPLAY="${DISPLAY:-:0}"
if [ -z "${XAUTHORITY:-}" ] && [ -f "${HOME}/.Xauthority" ]; then
  export XAUTHORITY="${HOME}/.Xauthority"
fi

cd "${SCRIPT_DIR}"
echo "[start] Bird_ws=${BIRD_WS}"
echo "[start] 手势1=撒娇 2~4=动作库 (稳定2s触发, 预览加 --preview)"
exec python3 hand_follow_robot.py --fast "$@"
