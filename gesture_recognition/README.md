# gesture_recognition — 手势识别与动作

ZED + MediaPipe 识别手势 **0~5**、手掌 3D 位置与移动方向；**脸部跟踪**常开（手势 **5** 确认后自动关闭）；手势 **1~4** 稳定 2s 后触发机器人动作；手势 **5** 稳定 **8s** 后切换 `start_hand_tracking.sh`；跟手中 **8s** 无五指则切回本程序。

## 一键启动

```bash
cd ~/Bird_ws/hand_identify/gesture_recognition
chmod +x start.sh
./start.sh
```

或从工程根目录：

```bash
./start_gesture_recognition.sh

若报 `CAMERA STREAM FAILED TO START`：多为 ZED 已被占用（勿用 Ctrl+Z 挂起程序）。执行 `pkill -f zed_gesture_recognition.py` 或 `./start_gesture_recognition.sh --force`。
```

## 常用参数

| 参数 | 说明 |
|------|------|
| `--preview` | 仅识别与日志，不执行手势动作 |
| `--no-gui` | 无窗口 |
| `--no-face-track` | 禁用脸部跟踪 |
| `--fast` | 更激进降频(proc 480) |
| `--hd1080` | 使用 1080p 采集（默认 720p） |

默认已优化：手势优先、隔帧人脸、动作失败自动重试、丢手 0.45s 内不计入复位。
| `--full-res-gui` | 1080p 全分辨率显示(更卡) |
| `--no-coquette` | 禁用手势 1 撒娇扭腰 |
| `--no-actions` | 禁用全部动作 |
| `--no-fsm` | 不等待 FSM=5 |
| `--gesture-hold-sec 2` | 触发前稳定时长 |

终端 **Ctrl+C** 可强制退出（`start.sh` 会转发信号；连按两次可 SIGKILL）。

## 文件

| 文件 | 说明 |
|------|------|
| `start.sh` | 一键启动 |
| `zed_gesture_recognition.py` | 主程序 |
| `face_tracker.py` | 内嵌脸跟踪（locate_face 控制律） |
| `gesture_motion.py` | ROS 动作调度 |
| `motion/hand_action_library.py` | 手势 2~4 → `/joy_msg` |
| `motion/waist_coquette_*.py` | 手势 1 撒娇扭腰（匀速连续、端点不停） |
| `handoff.py` | 手势 5 → `start_hand_tracking.sh` |

公共模块见 `../common/`。
