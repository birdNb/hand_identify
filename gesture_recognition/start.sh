#!/usr/bin/env bash
# 一键启动：ZED 手势识别 0~5 + 手势 1~4 机器人动作（需 ROS / sim2real）
# Ctrl+C 会转发 SIGINT 给 Python；若未退出可再按一次 Ctrl+C
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck source=/dev/null
source "${ROOT}/common/ros_env.sh"

cd "${SCRIPT_DIR}"
echo "[gesture_recognition] 目录: ${SCRIPT_DIR}"
echo "[gesture_recognition] 脸跟踪常开 | 手势 0急停 1撒娇 2~4动作 5→手部跟踪(无GUI)"
echo "[gesture_recognition] ESC 退出窗口 | Ctrl+C 强制退出终端"
echo "[gesture_recognition] 仅预览请加参数: --preview"

PY_ARGS=()
FORCE=0
for a in "$@"; do
  if [[ "$a" == "--force" ]]; then
    FORCE=1
  else
    PY_ARGS+=("$a")
  fi
done

if pgrep -f 'zed_gesture_recognition.py' >/dev/null 2>&1; then
  echo "[gesture_recognition] 警告: 已有 zed_gesture_recognition 在运行或挂起(Ctrl+Z)，ZED 无法二次打开"
  echo "  处理: pkill -f zed_gesture_recognition.py"
  echo "  或: fg 到前台后 Esc 退出窗口 / Ctrl+C"
  if [[ "${FORCE}" -eq 0 ]]; then
    exit 1
  fi
  echo "[gesture_recognition] --force: 结束旧进程后继续..."
  pkill -f 'zed_gesture_recognition.py' 2>/dev/null || true
  sleep 1
fi

CHILD_PID=""
_cleanup() {
  if [[ -n "${CHILD_PID}" ]] && kill -0 "${CHILD_PID}" 2>/dev/null; then
    echo ""
    echo "[gesture_recognition] 正在结束进程 (PID ${CHILD_PID})..."
    kill -INT "${CHILD_PID}" 2>/dev/null || true
    local i
    for i in 1 2 3 4 5 6 10; do
      kill -0 "${CHILD_PID}" 2>/dev/null || break
      sleep 0.2
    done
    if kill -0 "${CHILD_PID}" 2>/dev/null; then
      echo "[gesture_recognition] 发送 SIGTERM..."
      kill -TERM "${CHILD_PID}" 2>/dev/null || true
      sleep 0.3
      kill -KILL "${CHILD_PID}" 2>/dev/null || true
    fi
    wait "${CHILD_PID}" 2>/dev/null || true
  fi
  exit 130
}
trap _cleanup INT TERM

python3 "${SCRIPT_DIR}/zed_gesture_recognition.py" "${PY_ARGS[@]}" &
CHILD_PID=$!
wait "${CHILD_PID}"
exit $?
