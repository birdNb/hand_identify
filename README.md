# hand_identify — ZED Mini 手势与 3D 跟踪

## 功能

- MediaPipe Hands 检测单手 21 关键点
- 手势数字识别 **0~5**（伸直手指数）
- ZED 深度/点云计算手掌中心 **3D 坐标**（米）
- 帧间位移判断移动方向：**前/后/左/右/上/下**
- OpenCV 画面叠加 + 终端彩色日志

## 依赖

1. 安装 [ZED SDK](https://www.stereolabs.com/developers/release/)（建议 4.2+）
2. Python 包：

```bash
pip install -r requirements.txt
# ZED SDK 安装后:
pip install pyzed
```

## 子工程 hand_follow（手势 5 跟手）

手势 **5** 时底盘跟随手部距离与横向位置，输出 `/cmd_vel`。详见 [hand_follow/README.md](hand_follow/README.md)。

```bash
cd hand_identify/hand_follow
python3 hand_follow_robot.py              # 调试
python3 hand_follow_robot.py --enable-motion
```

## 运行（仅识别）

```bash
cd hand_identify
python3 zed_gesture_recognition.py
```

无界面（仅终端日志）：

```bash
python3 zed_gesture_recognition.py --no-gui
```

按 **ESC** 退出，**f** 切换全屏（默认自动全屏）。

## 分辨率 (对齐 locate_face)

- 默认 **ZED SDK HD1080**（1920×1080 @ 30fps），比 HD720 更清晰
- 显示与深度/点云用**全分辨率**；MediaPipe 输入缩至 **960px 宽**（与 `locate_face` 的 `PROC_MAX_W` 一致）
- 若性能不足: `--hd720` 或 `--proc-max-w 720`

## 参数

| 参数 | 说明 |
|------|------|
| `--no-gui` | 不弹窗 |
| `--move-threshold 0.03` | 移动判定阈值(米)，默认 0.02 |
| `--hd720` | 降为 HD720 采集 |
| `--proc-max-w 960` | MediaPipe 最大输入宽度 |
| `--dist-min 0.2` | 最近识别距离(米) |
| `--dist-max 2.0` | 最远识别距离(米) |

## 调试

- 深度不准：在脚本里把 `DEPTH_MODE.QUALITY` 改为 `ULTRA`
- 手势不稳：提高 `min_detection_confidence`（默认 0.7）；或调大 `THUMB_EXTEND_MIN`（减少拇指误计）
- 1 被识别成 2：拇指误检，可调 `THUMB_EXTEND_MIN`
- 5 被识别成 3/4：无名指/小指漏检，已改「指尖-手腕距离」判定 + 小指略放宽
- 方向太敏感：增大 `--move-threshold`
- 识别距离：默认 **0.2~2.0m**（Z 深度），超出范围只显示骨架、不计手势
