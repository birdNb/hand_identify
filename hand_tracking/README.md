# hand_tracking — 手部跟踪

基于 `hand_perception` 的 ZED 感知：

1. **左右居中**：掌心在画面左右偏移超过中心 **20%**（`|dx_norm| > 0.2`）时，发布 `/cmd_vel.angular.z = ±1.5`（强驱动），使手保持在机器人正前方。
2. **前后距离**（手势 **5**）：进入跟随后且识别为手势 5 时，另发布 `/cmd_vel.linear.x = ±0.5` 做距离保持（目标约 0.5m）。

## 一键启动

```bash
cd ~/Bird_ws/hand_identify/hand_tracking
chmod +x start.sh
./start.sh
```

或从工程根目录：

```bash
./start_hand_tracking.sh
```

默认 `--no-fsm`；需要 FSM 守门时去掉该参数或编辑 `start.sh`。

监听物理手柄 `/joy`：有摇杆/按键输入时，**5 秒内**手部跟踪**完全不发布** `/cmd_vel`（不发送零速，避免覆盖 `joy.yaml` teleop）。仅在需要跟手且非零速度时才发布，松手后只发一次零速收尾。

## 常用参数

| 参数 | 说明 |
|------|------|
| `--no-gui` | 无窗口（`start.sh` 未默认加，可自行追加） |
| `--no-fsm` | 不等待 FSM=EXEC_DEFAULT |
| `--dry-run` | 不发 `/cmd_vel`，只打印 |
| `--no-joy` | 不监听 `/joy` 手柄仲裁 |
| `--dist-min 0.2` | 最近有效距离 |
| `--dist-max 2.0` | 最远有效距离 |

## 文件

| 文件 | 说明 |
|------|------|
| `start.sh` | 一键启动 → `distance_hold.py` |
| `hand_perception.py` | 感知库（可单独 `python3 hand_perception.py` 预览） |
| `distance_hold.py` | 手部跟踪主程序（左右转 + 手势5距离） |
| `locomotion.py` | 全轴跟手备份（未接入 start.sh） |

公共模块见 `../common/`（`ros_control.py` 提供 FSM 监听）。
