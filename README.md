# 井字井字棋（Ultimate Tic-Tac-Toe）

这是一个面向 macOS 用户的井字井字棋（Ultimate Tic-Tac-Toe）桌面游戏，实现了以下特性：

- 9×9 棋盘，由九个 3×3 小棋盘组成。
- 小棋盘内部规则与经典井字棋相同，先连成一线即获胜，占领该小棋盘。
- 大棋盘胜利条件：在小棋盘的胜负记录上再连成一线。
- AlphaZero 风格的强化学习 AI，支持三个难度等级：简单 / 中等 / 困难。
- 内置 Tkinter 图形界面，开箱即玩，可在本地离线运行。
- 落子区域实时高亮，并对无效点击给出即时反馈。
- 提供快速训练入口，以及命令行批量训练脚本。

## 运行环境

- Python 3.10+
- macOS 自带的 Tk (推荐安装 [python.org](https://www.python.org/downloads/mac-osx/) 发行版，已包含 Tkinter)

## 快速开始

```bash
python -m ultimate_ttt.gui
```

界面启动后即可选择玩家棋子和难度等级。若选择让 AI 先手，请将玩家棋子设为 `O` 并重新开始。

## 游戏规则提示

- 每步落子后，下一步的可落子小棋盘由上一手所在的小格决定。
- 若目标小棋盘已被占领或填满，玩家可以在任意未结束的小棋盘落子。
- 界面会以绿色高亮提示本回合允许落子的区域。

## AI 难度

- **Easy（简单）**：随机合法落子。
- **Medium（中等）**：优先寻找即将获胜的位置，再尝试阻挡对手，否则使用带探索的策略。
- **Hard（困难）**：使用训练好的强化学习策略，仅保留少量探索。

## 强化学习训练

 图形界面中的“训练 AI”按钮会运行 200 局快速自我对弈，并自动保存策略。你也可以通过命令行批量训练：

```bash
python -m ultimate_ttt.train --episodes 5000 --simulations 200
```

可选参数：

- `--model-path`: 保存神经网络权重的 JSON 文件路径，默认位于 `ultimate_ttt/models/ultimate_ttt_alpha.json`。
- `--simulations`: 每一步使用的 MCTS 搜索次数。
- `--replay-size` / `--batch-size`: 自对弈经验回放池大小与训练批量。
- `--learning-rate` / `--training-steps`: 神经网络优化器相关参数。
- `--seed`: 设定随机种子以复现训练过程。

训练完成后重新启动 GUI 即可加载最新策略。

## 项目结构

```
ultimate_ttt/
├── __init__.py          # 包导出
├── ai.py                # 强化学习策略与辅助函数
├── game.py              # 核心棋盘逻辑
├── gui.py               # Tkinter 图形界面
├── models/
│   └── ultimate_ttt_alpha.json  # AlphaZero 网络参数
└── train.py             # 自我对弈训练脚本
```

## 开源许可

本项目可在本地自由使用与修改。
