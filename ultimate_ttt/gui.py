"""Tkinter-based graphical client for the Ultimate Tic-Tac-Toe game."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

from .ai import (
    UltimateTTTRLAI,
    block_opponent_move,
    immediate_winning_move,
)
from .game import InvalidMoveError, Move, UltimateTicTacToe

MODEL_PATH = Path(__file__).resolve().parent / "models" / "ultimate_ttt_alpha.json"


class UltimateTTTApp:
    BOARD_SIZE = 540
    PADDING = 20
    CELL_SIZE = BOARD_SIZE / 9

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("井字井字棋 - Ultimate Tic-Tac-Toe")
        self.root.resizable(False, False)

        self.agent = UltimateTTTRLAI.load(str(MODEL_PATH))
        self.game = UltimateTicTacToe()
        self.human_player = tk.StringVar(value="X")
        self.difficulty = tk.StringVar(value="Hard")

        self.status_var = tk.StringVar()
        self.hover_move: Optional[Move] = None
        self.invalid_move: Optional[Move] = None
        self._invalid_job: Optional[str] = None
        self._build_widgets()
        self.update_status()
        self.draw_board()

        if self.human_player.get() == "O":
            self.root.after(300, self.perform_ai_move)

    def _build_widgets(self) -> None:
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(control_frame, text="玩家棋子:").pack(side=tk.LEFT)
        human_menu = ttk.OptionMenu(
            control_frame,
            self.human_player,
            self.human_player.get(),
            "X",
            "O",
            command=lambda _: self.start_new_game(),
        )
        human_menu.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="难度:").pack(side=tk.LEFT)
        difficulty_menu = ttk.OptionMenu(
            control_frame,
            self.difficulty,
            self.difficulty.get(),
            "Easy",
            "Medium",
            "Hard",
        )
        difficulty_menu.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            control_frame,
            text="重新开始",
            command=self.start_new_game,
        ).pack(side=tk.LEFT, padx=10)

        ttk.Button(
            control_frame,
            text="训练AI",
            command=self.open_training_dialog,
        ).pack(side=tk.LEFT)

        ttk.Label(control_frame, textvariable=self.status_var).pack(
            side=tk.RIGHT
        )

        canvas_size = self.BOARD_SIZE + 2 * self.PADDING
        self.canvas = tk.Canvas(
            self.root,
            width=canvas_size,
            height=canvas_size,
            background="#f8f8f8",
        )
        self.canvas.pack(padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Motion>", self.on_motion)
        self.canvas.bind("<Leave>", self.on_leave_canvas)

    def start_new_game(self) -> None:
        self.game.reset()
        self.agent = UltimateTTTRLAI.load(str(MODEL_PATH))
        self.draw_board()
        self.update_status()
        self.hover_move = None
        self.invalid_move = None
        if self._invalid_job is not None:
            self.root.after_cancel(self._invalid_job)
            self._invalid_job = None
        if self.human_player.get() == "O":
            self.root.after(300, self.perform_ai_move)

    def open_training_dialog(self) -> None:
        if messagebox.askyesno("训练AI", "立即运行一次快速训练? (约200局)"):
            from .train import train_agent

            train_agent(episodes=200, model_path=MODEL_PATH)
            self.agent = UltimateTTTRLAI.load(str(MODEL_PATH))
            messagebox.showinfo("训练完成", "AI 模型已更新。")

    def on_click(self, event: tk.Event) -> None:
        if self.game.terminal:
            return
        if self.game.active_player() != self.human_player.get():
            return

        move = self._point_to_move(event.x, event.y)
        if move is None:
            return

        try:
            self.game.make_move(self.human_player.get(), move)
        except InvalidMoveError:
            self.show_invalid_feedback(move)
            return

        self.draw_board()
        self.update_status()
        if not self.game.terminal:
            self.root.after(200, self.perform_ai_move)

    def on_motion(self, event: tk.Event) -> None:
        if self.game.terminal:
            return
        move = self._point_to_move(event.x, event.y)
        valid_moves = set(self.game.available_moves())
        if (
            move is not None
            and move in valid_moves
            and self.game.active_player() == self.human_player.get()
        ):
            if move != self.hover_move:
                self.hover_move = move
                self.draw_board()
            self.canvas.configure(cursor="hand2")
        else:
            if self.hover_move is not None:
                self.hover_move = None
                self.draw_board()
            self.canvas.configure(cursor="arrow")

    def on_leave_canvas(self, _event: tk.Event) -> None:
        if self.hover_move is not None:
            self.hover_move = None
            self.draw_board()
        self.canvas.configure(cursor="arrow")

    def show_invalid_feedback(self, move: Move) -> None:
        self.invalid_move = move
        self.draw_board()
        self.status_var.set("该位置不可落子，请点击高亮区域。")
        if self._invalid_job is not None:
            self.root.after_cancel(self._invalid_job)
        self._invalid_job = self.root.after(800, self.clear_invalid_feedback)

    def clear_invalid_feedback(self) -> None:
        self.invalid_move = None
        self._invalid_job = None
        self.update_status()
        self.draw_board()

    def perform_ai_move(self) -> None:
        if self.game.terminal:
            return
        ai_player = "O" if self.human_player.get() == "X" else "X"
        if self.game.active_player() != ai_player:
            return

        move = self.choose_ai_move(ai_player)
        if move is None:
            return
        self.game.make_move(ai_player, move)
        self.draw_board()
        self.update_status()

    def choose_ai_move(self, player: str) -> Optional[Move]:
        moves = self.game.available_moves()
        if not moves:
            return None
        difficulty = self.difficulty.get()
        if difficulty == "Easy":
            return random.choice(moves)
        if difficulty == "Medium":
            winning = immediate_winning_move(self.game, player)
            if winning:
                return winning
            block = block_opponent_move(self.game, player)
            if block:
                return block
            self.agent.set_num_simulations(80)
            return self.agent.select_move(self.game, player, temperature=0.2)
        self.agent.set_num_simulations(160)
        return self.agent.select_move(self.game, player, temperature=0.0)

    def update_status(self) -> None:
        if self.game.terminal:
            if self.game.winner:
                self.status_var.set(f"{self.game.winner} 获胜!")
            else:
                self.status_var.set("平局")
            return

        forced = self.game.highlight_boards()
        if len(forced) == 1:
            text = f"请在第 {forced[0] + 1} 个小棋盘下子"
        else:
            text = "可以自由选择落子区域"
        self.status_var.set(
            f"轮到 {self.game.active_player()} ({text})"
        )

    def draw_board(self) -> None:
        self.canvas.delete("all")
        margin = self.PADDING
        size = self.BOARD_SIZE
        cell = self.CELL_SIZE

        forced = set(self.game.highlight_boards())
        for sub_idx in range(9):
            top_row, top_col = divmod(sub_idx, 3)
            x0 = margin + top_col * cell * 3
            y0 = margin + top_row * cell * 3
            x1 = x0 + cell * 3
            y1 = y0 + cell * 3

            fill = "#ffffff"
            if self.game.macro_owner(sub_idx) == "X":
                fill = "#cce1ff"
            elif self.game.macro_owner(sub_idx) == "O":
                fill = "#ffd6cc"
            elif sub_idx in forced:
                fill = "#f2fce4"

            self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="")

            for idx, value in enumerate(self.game.cells(sub_idx)):
                row, col = divmod(idx, 3)
                cx = x0 + col * cell + cell / 2
                cy = y0 + row * cell + cell / 2
                if value == "X":
                    offset = cell * 0.3
                    self.canvas.create_line(
                        cx - offset,
                        cy - offset,
                        cx + offset,
                        cy + offset,
                        width=3,
                        fill="#1a4b8c",
                    )
                    self.canvas.create_line(
                        cx - offset,
                        cy + offset,
                        cx + offset,
                        cy - offset,
                        width=3,
                        fill="#1a4b8c",
                    )
                elif value == "O":
                    radius = cell * 0.35
                    self.canvas.create_oval(
                        cx - radius,
                        cy - radius,
                        cx + radius,
                        cy + radius,
                        width=3,
                        outline="#b53d00",
                    )

        # Draw grid lines
        for i in range(10):
            line_width = 1 if i % 3 else 4
            start = margin + i * cell
            self.canvas.create_line(
                margin,
                start,
                margin + size,
                start,
                width=line_width,
                fill="#444444",
            )
            self.canvas.create_line(
                start,
                margin,
                start,
                margin + size,
                width=line_width,
                fill="#444444",
            )

        # Highlight last move for clarity
        if self.game.last_move is not None:
            x0, y0, x1, y1 = self._cell_bbox(self.game.last_move)
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#f1a208",
                width=3,
            )

        # Hover highlight for pending human move
        if (
            self.hover_move is not None
            and not self.game.terminal
            and self.hover_move in self.game.available_moves()
            and self.game.active_player() == self.human_player.get()
        ):
            x0, y0, x1, y1 = self._cell_bbox(self.hover_move)
            self.canvas.create_rectangle(
                x0 + 2,
                y0 + 2,
                x1 - 2,
                y1 - 2,
                outline="#4caf50",
                width=3,
            )

        # Temporary feedback for invalid click attempts
        if self.invalid_move is not None:
            x0, y0, x1, y1 = self._cell_bbox(self.invalid_move)
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#d32f2f",
                width=3,
            )
            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                fill="#d32f2f",
                stipple="gray25",
                outline="",
            )

    def _point_to_move(self, x: float, y: float) -> Optional[Move]:
        rel_x = x - self.PADDING
        rel_y = y - self.PADDING
        if rel_x < 0 or rel_y < 0 or rel_x >= self.BOARD_SIZE or rel_y >= self.BOARD_SIZE:
            return None

        grid_x = int(rel_x // self.CELL_SIZE)
        grid_y = int(rel_y // self.CELL_SIZE)
        sub_row, cell_row = divmod(grid_y, 3)
        sub_col, cell_col = divmod(grid_x, 3)
        sub_index = sub_row * 3 + sub_col
        cell_index = cell_row * 3 + cell_col
        return (sub_index, cell_index)

    def _cell_bbox(self, move: Move) -> tuple[float, float, float, float]:
        sub_index, cell_index = move
        cell = self.CELL_SIZE
        top_row, top_col = divmod(sub_index, 3)
        cell_row, cell_col = divmod(cell_index, 3)
        x0 = self.PADDING + top_col * cell * 3 + cell_col * cell
        y0 = self.PADDING + top_row * cell * 3 + cell_row * cell
        x1 = x0 + cell
        y1 = y0 + cell
        return x0, y0, x1, y1

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = UltimateTTTApp()
    app.run()


if __name__ == "__main__":
    main()
