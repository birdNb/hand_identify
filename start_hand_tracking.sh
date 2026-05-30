#!/usr/bin/env bash
# =============================================================================
# 手部跟踪 — 操作说明（start_hand_tracking.sh）
# =============================================================================
#
# 【前置条件 · 必做】
#   本功能仅在机器人已启动 RL 步态策略、且处于「默认步态执行」时可用：
#     - 先正常启动 sim2real（含 RL policy / walk 步态），完成上电与初始化；
#     - FSM 须进入 EXEC_DEFAULT（/fsm_state == 5），即默认 RL 步态策略运行模式；
#     - 未进入该模式时，/cmd_vel 可能被底层忽略或行为异常，请勿启动本脚本。
#   可用下面命令确认（另开终端）：
#     rostopic echo /fsm_state -n 1    # 应看到 data: 5
#
# 【依赖】
#   - ROS 已 source（由 hand_tracking/start.sh 内 ros_env.sh 处理）
#   - ZED Mini 相机可用
#   - 底盘 teleop 通过 joy.yaml 发布 /cmd_vel（与手部跟踪共用该话题）
#
# 【启动】
#   cd ~/Bird_ws/hand_identify
#   ./start_hand_tracking.sh              # 默认无 GUI
#   ./start_hand_tracking.sh --gui        # 带画面调试
#   ./start_hand_tracking.sh --dry-run    # 只打印，不发速度
#
# 【功能】
#   1. 左右居中：手掌在画面左右偏移超过中心 20% 时，发布 angular.z = ±1.5；
#   2. 前后距离：识别手势 5 后，另发布 linear.x = ±0.5（目标距离约 0.5 m）；
#   3. 手柄优先：/joy 有摇杆/按键输入后 5 秒内，手部跟踪不发布 /cmd_vel，避免盖手柄。
#
# 【注意】
#   - 勿与 start_gesture_recognition.sh 同时运行（会抢 /cmd_vel）。
#   - 默认带 --no-fsm（见 hand_tracking/start.sh）；若需程序内等待 FSM=5，
#     可编辑 hand_tracking/start.sh 去掉 --no-fsm，或自行调用 distance_hold.py。
#   - ESC（GUI 模式）或 Ctrl+C 退出；退出前若曾发过速度会补发一次零速收尾。
#
# =============================================================================

set -euo pipefail

print_usage() {
    cat <<'EOF'
[手部跟踪] 使用前提：机器人须在 RL 步态策略下，且 FSM=EXEC_DEFAULT(5)。

启动:  ./start_hand_tracking.sh [--gui] [--dry-run] [--no-joy] ...
确认:  rostopic echo /fsm_state -n 1   # 期望 data: 5

功能:  手掌左右偏移 → 底盘转向；手势 5 → 前后距离保持；手柄输入后 5s 内让路。
EOF
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${1:-}" in
    -h|--help)
        print_usage
        exit 0
        ;;
esac

echo "[手部跟踪] 请确认 sim2real 已运行且 FSM=EXEC_DEFAULT(5)（RL 默认步态策略）"
print_usage
echo ""

exec "${ROOT}/hand_tracking/start.sh" "$@"
