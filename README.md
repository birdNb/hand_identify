# hand_identify — ZED Mini 手部感知

两大功能模块，共用 `common/` 下的手势定义与 ROS 工具。

## 目录结构

```
hand_identify/
├── start_gesture_recognition.sh   # 根目录快捷启动 → 手势识别
├── start_hand_tracking.sh         # 根目录快捷启动 → 手部跟踪
├── requirements.txt
├── common/                        # 公共
│   ├── gesture_actions.py         # 手势 0~5 定义、稳定判定
│   ├── ros_setup.py               # sim2real Python 路径
│   ├── ros_control.py             # FSM / 手柄仲裁
│   ├── paths.py                   # 模块路径
│   └── ros_env.sh                 # source ROS 环境
├── gesture_recognition/           # 手势识别 + 脸跟踪 + 动作
│   ├── start.sh
│   ├── zed_gesture_recognition.py
│   ├── face_tracker.py            # locate_face 同律，共用 ZED
│   ├── gesture_motion.py
│   └── motion/                    # /joy_msg、撒娇扭腰
└── hand_tracking/                 # 手部跟踪（底盘跟手）
    ├── start.sh
    ├── hand_perception.py         # ZED 感知库
    ├── distance_hold.py           # 左右居中 + 手势5 距离保持
    └── locomotion.py              # 全轴跟手备份
```

## 依赖

```bash
pip install -r requirements.txt
pip install pyzed   # 安装 ZED SDK 后
```

## 一键启动

### 手势识别（0~5 + 脸跟踪 + 动作 1~4）

```bash
cd ~/Bird_ws/hand_identify
./start_gesture_recognition.sh
```

仅识别、不控机器人：`./start_gesture_recognition.sh --preview`

详见 [gesture_recognition/README.md](gesture_recognition/README.md)

### 手部跟踪（左右居中 + 手势 5 距离）

```bash
cd ~/Bird_ws/hand_identify
./start_hand_tracking.sh
```

详见 [hand_tracking/README.md](hand_tracking/README.md)

## 手势一览

| 手势 | 手势识别 | 手部跟踪 |
|------|----------|----------|
| 0 | 急停 / 按住 5s 退出 | — |
| 1 | 撒娇扭腰 | — |
| 2~4 | 抬手 / 挥双手 / 踢球 | — |
| 5 | 稳定 **8s** ↔ 跟手；跟手 **8s** 无五指切回 | 左右转 + 手势5 前后距离 → `/cmd_vel` |
