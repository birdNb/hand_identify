#!/usr/bin/env bash
# 统一启动：手势跟手 + 默认脸部跟踪(locate_face) + 识别到手抬头反馈
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
echo "[start] 运行 hand_follow_robot (HD720+640px 默认, 预览加 --preview)"
exec python3 hand_follow_robot.py --fast "$@"
