#!/usr/bin/env bash
# 独立启动：五指(5)仅前后距离保持（目标40cm，手势稳定2s后进入）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
echo "[start] 5指距离保持独立调试（默认无GUI+跳过FSM，仅测相机/距离）"
exec python3 "${SCRIPT_DIR}/5_finger_distance_hold_debug.py" --no-fsm "$@"
