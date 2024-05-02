from __future__ import annotations
from typing import Optional, Callable
from ..config import Config
from enum import Enum
import numpy as np


class Player(Enum):
    PLAYER1 = 1
    PLAYER2 = 2


class Game:

    def __init__(self, config: Config, player_controllers: dict[str, Callable]) -> None:
        self.config = config
        self.player_controllers = player_controllers

        self.row_number = config.game.row_number
        self.column_number = config.game.column_number
        self.to_align = config.game.to_align

        self.board = np.zeros((self.row_number, self.column_number))
        self.current_player = Player.PLAYER1
        self.winner = None

        # row, column, diagonal, anti-diagonal
        self.open_lines = {
            Player.PLAYER1: np.zeros((4, self.row_number, self.column_number)),
            Player.PLAYER2: np.zeros((4, self.row_number, self.column_number)),
        }
        self.open_lines_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}

        self.move_history = []

    # Request
    def get_current_player(self) -> Player:
        """Return the current player."""
        return self.current_player

    def get_board(self) -> np.ndarray:
        """Return the current board."""
        return self.board

    def get_winner(self) -> Optional[Player]:
        """Return the winner if there is one, else return None."""
        return self.winner

    def get_possible_moves(self) -> list[tuple[int, int]]:
        """Return the possible moves."""
        return [
            (i, j)
            for i in range(self.row_number)
            for j in range(self.column_number)
            if self.board[i, j] == 0
        ]

    def get_move_history(self) -> list[tuple[int, int]]:
        """Return the move history."""
        return self.move_history

    def get_number_of_open_lines(self, player: Player) -> int:
        """Return the number of open lines for the given player."""
        return self.open_lines_count[player]

    def check_winner(self, row: int, column: int, player: Player) -> bool:
        """Check if the given player has won the game in O(n)."""
        # list all the rows that could be affected by the move
        for i in range(self.to_align):
            start = max(0, column - self.to_align + 1 + i)
            end = min(self.column_number, column + 1 + i)
            if end - start != self.to_align:
                continue
            if np.all(self.board[row, start:end] == player.value):
                return True

        # list all the columns that could be affected by the move
        for i in range(self.to_align):
            start = max(0, row - self.to_align + 1 + i)
            end = min(self.row_number, row + 1 + i)
            if end - start != self.to_align:
                continue
            if np.all(self.board[start:end, column] == player.value):
                return True

        # list all the diagonals that could be affected by the move
        for i in range(self.to_align):
            start_row = max(0, row - self.to_align + 1 + i)
            start_column = max(0, column - self.to_align + 1 + i)
            end_row = min(self.row_number, row + 1 + i)
            end_column = min(self.column_number, column + 1 + i)
            if (
                end_row - start_row != self.to_align
                or end_column - start_column != self.to_align
            ):
                continue
            if np.all(
                np.diag(self.board[start_row:end_row, start_column:end_column])
                == player.value
            ):
                return True

        # list all the anti-diagonals that could be affected by the move
        for i in range(self.to_align):
            start_row = max(0, row - i)
            start_column = max(0, column - self.to_align + 1 + i)
            end_row = min(self.row_number, row + self.to_align - i)
            end_column = min(self.column_number, column + i + 1)
            if (
                end_row - start_row != self.to_align
                or end_column - start_column != self.to_align
            ):
                continue
            if np.all(
                np.diag(
                    np.fliplr(self.board[start_row:end_row, start_column:end_column])
                )
                == player.value
            ):
                return True

        return False

    def is_full(self) -> bool:
        """Check if the board is full."""
        return bool(np.all(self.board != 0))

    def is_over(self) -> bool:
        """Check if the game is over."""
        return self.get_winner() is not None or self.is_full()

    def get_value(self, player: Player) -> float:
        """Return the value of the game for the given player."""
        winner = self.get_winner()
        if winner is None:
            return 0
        if winner == player:
            return 1
        return -1

    def is_valid_move(self, row: int, column: int) -> bool:
        """Check if the move is valid."""
        return self.board[row, column] == 0

    # Commands
    def play(self, row: int, column: int) -> None:
        """
        Play a move at the given row and column for the current player.
        Switch the player afterwards.
        """
        if not self.is_valid_move(row, column):
            raise ValueError("Invalid move")

        self.board[row, column] = self.current_player.value
        if self.check_winner(row, column, self.current_player):
            self.winner = self.current_player
        self.switch_player()
        self.update_open_lines(row, column, self.current_player)
        self.move_history.append((row, column))

    def update(self) -> None:
        """Update the game state."""
        if self.is_over():
            return
        move = self.player_controllers[self.current_player.name](self)
        if move is None:
            return
        self.play(*move)

    def undo(self) -> None:
        """Undo the last move."""
        if not self.move_history or self.is_over():
            return
        row, column = self.move_history.pop()
        self.board[row, column] = 0
        self.current_player = Player.PLAYER1

        # Reset the open lines
        self.open_lines_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}
        self.open_lines = {
            Player.PLAYER1: np.zeros((4, self.row_number, self.column_number)),
            Player.PLAYER2: np.zeros((4, self.row_number, self.column_number)),
        }
        self.init_open_lines()
        for move in self.move_history:
            self.update_open_lines(move[0], move[1], self.get_opponent(self.current_player))
            self.switch_player()

    def init_open_lines(self) -> None:
        """
        Initialize the open lines for each player."""
        # Rows initialization
        for row in range(self.row_number):
            for col in range(self.column_number - self.to_align + 1):
                self.open_lines[Player.PLAYER1][0, row, col] = 1
                self.open_lines[Player.PLAYER2][0, row, col] = 1
                self.open_lines_count[Player.PLAYER1] += 1
                self.open_lines_count[Player.PLAYER2] += 1

        # Columns initialization
        for col in range(self.column_number):
            for row in range(self.row_number - self.to_align + 1):
                self.open_lines[Player.PLAYER1][1, row, col] = 1
                self.open_lines[Player.PLAYER2][1, row, col] = 1
                self.open_lines_count[Player.PLAYER1] += 1
                self.open_lines_count[Player.PLAYER2] += 1

        # Diagonals initialization
        for row in range(self.row_number - self.to_align + 1):
            for col in range(self.column_number - self.to_align + 1):
                self.open_lines[Player.PLAYER1][2, row, col] = 1
                self.open_lines[Player.PLAYER2][2, row, col] = 1
                self.open_lines_count[Player.PLAYER1] += 1
                self.open_lines_count[Player.PLAYER2] += 1

        # Anti-diagonals initialization
        for row in range(self.row_number - 1, self.to_align - 2, -1):
            for col in range(self.column_number - self.to_align + 1):
                self.open_lines[Player.PLAYER1][3, row, col] = 1
                self.open_lines[Player.PLAYER2][3, row, col] = 1
                self.open_lines_count[Player.PLAYER1] += 1
                self.open_lines_count[Player.PLAYER2] += 1

    # Utils
    def get_opponent(self, player: Player) -> Player:
        """Return the opponent of the current player."""
        match player:
            case Player.PLAYER1:
                return Player.PLAYER2
            case Player.PLAYER2:
                return Player.PLAYER1

    def switch_player(self) -> None:
        """Switch the current player."""
        self.current_player = self.get_opponent(self.current_player)

    def reset(self) -> None:
        """Reset the game."""
        self.board = np.zeros((self.row_number, self.column_number))
        self.current_player = Player.PLAYER1
        self.winner = None
        self.move_history = []
        self.open_lines = {
            Player.PLAYER1: np.zeros((4, self.row_number, self.column_number)),
            Player.PLAYER2: np.zeros((4, self.row_number, self.column_number)),
        }
        self.open_lines_count = {Player.PLAYER1: 0, Player.PLAYER2: 0}

    def copy(self) -> Game:
        """Return a copy of the current game."""
        game = Game(self.config, self.player_controllers)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.winner = self.winner
        game.move_history = self.move_history.copy()
        game.open_lines = {
            Player.PLAYER1: self.open_lines[Player.PLAYER1].copy(),
            Player.PLAYER2: self.open_lines[Player.PLAYER2].copy(),
        }
        game.open_lines_count = {
            Player.PLAYER1: self.open_lines_count[Player.PLAYER1],
            Player.PLAYER2: self.open_lines_count[Player.PLAYER2],
        }

        return game

    def update_open_lines(self, row: int, column: int, player: Player) -> None:
        """Update the open lines after a move for the given player."""
        # Rows update
        for i in range(max(0, column - self.to_align + 1), column + 1):
            cell_value = self.open_lines[player][0, row, i]
            if cell_value > 0:
                self.open_lines[player][0, row, i] = 0
                self.open_lines_count[player] -= 1

        # Columns update
        for i in range(max(0, row - self.to_align + 1), row + 1):
            cell_value = self.open_lines[player][1, i, column]
            if cell_value > 0:
                self.open_lines[player][1, i, column] = 0
                self.open_lines_count[player] -= 1

        # Diagonals update
        for i in range(max(0, row - self.to_align + 1), row + 1):
            for j in range(max(0, column - self.to_align + 1), column + 1):
                if row - i != column - j:
                    continue

                cell_value = self.open_lines[player][2, i, j]
                if cell_value > 0:
                    self.open_lines[player][2, i, j] = 0
                    self.open_lines_count[player] -= 1

        # Anti-diagonals update
        for i in range(min(self.row_number - 1, row + self.to_align - 1), row - 1, -1):
            for j in range(max(0, column - self.to_align + 1), column + 1):
                if row - i != j - column:
                    continue

                cell_value = self.open_lines[player][3, i, j]
                if cell_value > 0:
                    self.open_lines[player][3, i, j] = 0
                    self.open_lines_count[player] -= 1

    def get_value_and_terminated(self) -> tuple[int, bool]:
        """Return the value of the game and whether it is terminated after playing the given action."""
        is_over = self.is_over()
        if is_over:
            winner = self.get_winner()
            if winner is None:
                return 0, True
            return 1, True
        return 0, False
    
    def get_one_hot_valid_moves(self) -> np.ndarray:
        """Return the one-hot encoding of the valid moves."""
        return (self.board.reshape(-1) == 0).astype(np.uint8)
    
    def get_encoded_state(self) -> np.ndarray:
        """Return the encoded state of the game."""
        encoded_state = np.stack(
            (self.board == 0, self.board == 1, self.board == 2)
        ).astype(np.float32)
        
        return encoded_state
    
    def change_perspective(self, player: int) -> Game:
        """Change the perspective of the game."""
        if player == 1:
            return self
        new_board = np.zeros((self.row_number, self.column_number))
        new_board[self.board == 1] = 2
        new_board[self.board == 2] = 1

        new_open_lines = {
            Player.PLAYER1: self.open_lines[Player.PLAYER2].copy(),
            Player.PLAYER2: self.open_lines[Player.PLAYER1].copy(),
        }
        new_open_lines_count = {
            Player.PLAYER1: self.open_lines_count[Player.PLAYER2],
            Player.PLAYER2: self.open_lines_count[Player.PLAYER1],
        }

        new_game = Game(self.config, self.player_controllers)
        new_game.board = new_board
        new_game.current_player = self.get_opponent(self.current_player)
        new_game.winner = self.get_opponent(self.winner) if self.winner is not None else None
        new_game.move_history = self.move_history.copy()
        new_game.open_lines = new_open_lines
        new_game.open_lines_count = new_open_lines_count

        return new_game
