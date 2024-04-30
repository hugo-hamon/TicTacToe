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
        for player in Player:
            if self.check_winner(player):
                return player
        return None

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
    
    def better_check_winner(self, row: int, column: int, player: Player) -> bool:
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
            if end_row - start_row != self.to_align or end_column - start_column != self.to_align:
                continue
            if np.all(np.diag(self.board[start_row:end_row, start_column:end_column]) == player.value):
                return True

        # list all the anti-diagonals that could be affected by the move
        for i in range(self.to_align):
            start_row = max(0, row + self.to_align - 1 - i)
            start_column = max(0, column - self.to_align + 1 + i)
            end_row = min(self.row_number, row - i)
            end_column = min(self.column_number, column + 1 + i)
            print(start_row, end_row, start_column, end_column)
            if end_row - start_row != self.to_align or end_column - start_column != self.to_align:
                continue
            print(self.board[start_row:end_row, start_column:end_column])
            print(np.diag(np.fliplr(self.board[start_row:end_row, start_column:end_column])))
            if np.all(np.diag(np.fliplr(self.board[start_row:end_row, start_column:end_column])) == player.value):
                return True
            
        return False


    def check_winner(self, player: Player) -> bool:
        """Check if the given player has won the game."""
        # Check rows
        for i in range(self.row_number):
            for j in range(self.column_number - self.to_align + 1):
                if np.all(self.board[i, j : j + self.to_align] == player.value):
                    return True

        # Check columns
        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number):
                if np.all(self.board[i : i + self.to_align, j] == player.value):
                    return True

        # Check diagonals
        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number - self.to_align + 1):
                if np.all(
                    np.diag(self.board[i : i + self.to_align, j : j + self.to_align])
                    == player.value
                ):
                    return True
                if np.all(
                    np.diag(
                        np.fliplr(
                            self.board[i : i + self.to_align, j : j + self.to_align]
                        )
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
        self.better_check_winner(row, column, self.current_player)
        self.switch_player()
        self.move_history.append((row, column))

    def update(self) -> None:
        """Update the game state."""
        if self.is_over():
            return
        move = self.player_controllers[self.current_player.name](self)
        if move is None:
            return
        self.play(*move)

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

    def copy(self) -> Game:
        """Return a copy of the current game."""
        game = Game(self.config, self.player_controllers)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.move_history = self.move_history.copy()
        return game

    def line_open(self, player: Player) -> int:
        """Return the number of open lines for the given player."""
        count = 0
        opponent_value = self.get_opponent(player).value
        for i in range(self.row_number):
            for j in range(self.column_number - self.to_align + 1):
                if np.any(self.board[i, j : j + self.to_align] == opponent_value):
                    continue
                count += 1

        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number):
                if np.any(self.board[i : i + self.to_align, j] == opponent_value):
                    continue
                count += 1

        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number - self.to_align + 1):
                if np.any(
                    np.diag(self.board[i : i + self.to_align, j : j + self.to_align])
                    == opponent_value
                ):
                    continue
                count += 1
                if np.any(
                    np.diag(
                        np.fliplr(
                            self.board[i : i + self.to_align, j : j + self.to_align]
                        )
                    )
                    == opponent_value
                ):
                    continue
                count += 1

        return count
