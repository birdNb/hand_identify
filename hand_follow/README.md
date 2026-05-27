# hand_follow — 手势触发底盘跟手

将 `hand_perception` 输出的 **手势数字** 与 **掌心距离/横向偏差** 转为 `/cmd_vel` 控制指令。

## 控制映射

| 手势 | 动作 |
|------|------|
| **0** | **急停**：立即停止动作库与 `cmd_vel`；**持续按住 5s** 退出程序 |
| **1** | **脸部跟踪**：默认开（与手势**共用 ZED RGB**）；再比手势 1 关闭 |
| **2** | 抬手（`hello`，`rt+x`） |
| **3** | 挥动双手（`cheer`，`rt+a`） |
| **4** | 踢球（`byd_small_kick`，`x`） |
| **5** | 跟手：保持目标距离，横向对准；\|dx\| 大时优先旋转 |
| 无手 / 超距 | 无跟手指令；手势 0~4 需在有效距离内 |
| **手柄** | 检测到 `/joy` 有效输入 → 程序不发指令；摇杆 **空闲 3s** 后才恢复 |

手势 **1~4** 在识别到**上升沿**时经 `/joy_msg` 调用动作库（与手柄 `custom_action.yaml` 组合键一致），默认 **4s** 冷却防连发。执行动作期间不发 `cmd_vel`。

## 控制律（手势 5，且在 0.2~2 m 内）

指令 **±0.3 或 0**；位置误差 **|e| ≤ 0.3 m** 时该轴为 0。  
**手静止不发底盘指令**（帧间掌心位移 < 4 cm 视为不动）。  
**识别到手**：脖子在 **1 秒内**抬头 **10°** 后**停止**（反馈）；手离开后回中。脸跟踪开启时由脸跟踪控头，此项不生效。

| 指令 | cmd_vel | 误差来源 |
|------|---------|----------|
| **X** | `linear.x` | 深度 Z − 目标距离 |
| **Y** | `linear.y` | 掌心横向 `palm_x` |
| **Z** | `angular.z` | 掌心竖向 `palm_y` |
| 大横向 | 仅 **Z** | `\|palm_x\| > 0.5 m` 时不发 Y，只旋转 |

默认目标距离 **1.0 m**；阈值 **0.3 m**（`--pos-thresh`）。

## 依赖

```bash
cd ../..
pip install -r requirements.txt
pip install pyzed
```

ROS / sim2real（二选一）：

```bash
# 方式 A：脚本会自动把 ~/sim2real/install/.../dist-packages 加入路径
python3 hand_follow_robot.py --enable-motion

# 方式 B：先 source 再运行（推荐与 roslaunch 同终端）
source /opt/ros/noetic/setup.bash
source ~/sim2real/devel/setup.bash   # 或 install/setup.bash
python3 hand_follow_robot.py --enable-motion
```

## 运行（统一启动）

```bash
cd ~/Bird_ws/hand_identify/hand_follow
chmod +x start_hand_follow.sh

# 推荐：一键启动（自动 source ROS、默认开脸跟踪、默认发运动指令）
./start_hand_follow.sh

# 仅预览画面与日志（不发指令、不启 locate_face）
./start_hand_follow.sh --preview
```

或直接：

```bash
python3 hand_follow_robot.py          # 默认：运动 + 脸跟踪
python3 hand_follow_robot.py --preview
```

**控车条件（手势 5 跟手）**：非 `--preview` + FSM=5 + 无手柄占用 + 手势 5 + 有效距离 + 手在动。  
**脸跟踪**：与手势识别**共用一路 ZED 降采样 RGB**（不再单独开相机/子进程）；可用 `--no-face-track` 关闭。  
**动作 2~4**：通过 `/joy_msg` 触发。

- **ESC** 退出  
- **f** 切换全屏  
- `--no-fsm` 跳过 FSM 等待（仍建议在 EXEC_DEFAULT 下运行）  
- `--no-joy` 不启用手柄仲裁（调试用）  
- `--no-neck` 不控制脖子抬头  
- `--no-actions` 禁用手势 2~4 动作库  
- `--no-face-track` 不启脸部跟踪  
- `--no-face-track-auto` 不自动启脸跟踪（手势 1 仍可开）  
- `--preview` 预览，不发指令  
- `--zero-exit-sec 5` 手势 0 按住多少秒后退出（默认 5）  
- `--cmd-mag 0.3` 指令幅度  
- `--move-thresh 0.04` 判定手在动的位移(米)  
- `--target-dist 1.2` 跟手距离  
- `--dist-min 0.2 --dist-max 2.0` 识别有效距离  

## 文件

| 文件 | 说明 |
|------|------|
| `hand_follow_robot.py` | 主程序：感知 + 显示 + ROS 发布 |
| `hand_follow_control.py` | 控制律、VelCommand、FSM、cmd_vel 线程 |
| `hand_action_library.py` | 手势 1~4 → `/joy_msg` 动作库触发 |

感知复用上级目录 `hand_perception.py`（与 `zed_gesture_recognition.py` 同源算法）。
