#!/usr/bin/env bash
# 一键启动：手部左右居中(angular.z) + 手势5前后距离(linear.x)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${ROOT}/common/ros_env.sh"

cd "${SCRIPT_DIR}"
echo "[hand_tracking] 目录: ${SCRIPT_DIR}"
echo "[hand_tracking] 左右居中+手势5距离（默认 --no-fsm，可加 --no-gui）"
exec python3 "${SCRIPT_DIR}/distance_hold.py" --no-fsm "$@"
