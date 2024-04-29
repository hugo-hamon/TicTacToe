from __future__ import annotations
from typing import Optional
from enum import Enum
import numpy as np


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class Player(Enum):
    PLAYER1 = 1
    PLAYER2 = 2


def get_opponent(player: Player) -> Player:
    """Return the opponent of the given player."""
    return Player.PLAYER1 if player == Player.PLAYER2 else Player.PLAYER2


class TicTacToe:

    def __init__(self, row_number: int, column_number: int, to_align: int) -> None:
        self.row_number = row_number
        self.column_number = column_number
        self.to_align = to_align

        self.board = np.zeros((row_number, column_number))
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
        return [(i, j) for i in range(self.row_number) for j in range(self.column_number) if self.board[i, j] == 0]

    def check_winner(self, player: Player) -> bool:
        """Check if the given player has won the game."""
        # Check rows
        for i in range(self.row_number):
            for j in range(self.column_number - self.to_align + 1):
                if np.all(self.board[i, j:j+self.to_align] == player.value):
                    return True

        # Check columns
        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number):
                if np.all(self.board[i:i+self.to_align, j] == player.value):
                    return True

        # Check diagonals
        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number - self.to_align + 1):
                if np.all(np.diag(self.board[i:i+self.to_align, j:j+self.to_align]) == player.value):
                    return True
                if np.all(np.diag(np.fliplr(self.board[i:i+self.to_align, j:j+self.to_align])) == player.value):
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

    # Commands
    def play(self, row: int, column: int) -> None:
        """
        Play a move at the given row and column for the current player.
        Switch the player afterwards.
        """
        if self.board[row, column] != 0:
            raise ValueError("Invalid move")

        self.board[row, column] = self.current_player.value
        self.current_player = Player.PLAYER1 if self.current_player == Player.PLAYER2 else Player.PLAYER2
        self.move_history.append((row, column))

    # Utils
    def switch_player(self) -> None:
        """Switch the current player."""
        match self.current_player:
            case Player.PLAYER1:
                self.current_player = Player.PLAYER2
            case Player.PLAYER2:
                self.current_player = Player.PLAYER1

    def copy(self) -> TicTacToe:
        """Return a copy of the current game."""
        game = TicTacToe(self.row_number, self.column_number, self.to_align)
        game.board = self.board.copy()
        game.current_player = self.current_player
        game.move_history = self.move_history.copy()
        return game

    def display_game(self) -> None:
        """Display the current game."""
        current_player = self.get_current_player()
        for i in range(self.row_number):
            print(" ---" * self.column_number)
            for j in range(self.column_number):
                print("|", end=" ")
                if self.board[i, j] == 0:
                    print(" ", end=" ")
                elif self.board[i, j] == 1:
                    if self.move_history[-1] == (i, j):
                        print(bcolors.OKGREEN + "X" + bcolors.ENDC, end=" ")
                    else:
                        print("X", end=" ")
                else:
                    if self.move_history[-1] == (i, j):
                        print(bcolors.OKGREEN + "O" + bcolors.ENDC, end=" ")
                    else:
                        print("O", end=" ")
            print("|")
        print(" ---" * self.column_number)

    def line_open(self, player: Player) -> int:
        """Return the number of open lines for the given player."""
        count = 0
        opponent_value = get_opponent(player).value
        for i in range(self.row_number):
            for j in range(self.column_number - self.to_align + 1):
                if np.any(self.board[i, j:j+self.to_align] == opponent_value):
                    continue
                count += 1

        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number):
                if np.any(self.board[i:i+self.to_align, j] == opponent_value):
                    continue
                count += 1

        for i in range(self.row_number - self.to_align + 1):
            for j in range(self.column_number - self.to_align + 1):
                if np.any(np.diag(self.board[i:i+self.to_align, j:j+self.to_align]) == opponent_value):
                    continue
                count += 1
                if np.any(np.diag(np.fliplr(self.board[i:i+self.to_align, j:j+self.to_align])) == opponent_value):
                    continue
                count += 1

        return count


if __name__ == "__main__":
    game = TicTacToe(5, 5, 4)

    print(game.line_open(Player.PLAYER1))
