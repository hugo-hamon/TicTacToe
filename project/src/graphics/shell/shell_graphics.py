from ...game.game import Game
from ...config import Config


class bcolors:
    GREEN = "\033[92m"
    ENDC = "\033[0m"


class ShellGraphics:

    def __init__(self, config: Config) -> None:
        self.config = config

        self.row_number = config.game.row_number
        self.column_number = config.game.column_number
        self.player1_symbol = self.config.graphics.player1_symbol
        self.player2_symbol = self.config.graphics.player2_symbol

        if len(self.player1_symbol) != 1:
            raise ValueError("Player 1 symbol must be a single character.")
        if len(self.player2_symbol) != 1:
            raise ValueError("Player 2 symbol must be a single character.")

    def draw(self, game: Game) -> None:
        """Display the current game."""
        board = game.get_board()
        move_history = game.get_move_history()
        for i in range(self.row_number):
            print(" ---" * self.column_number)
            for j in range(self.column_number):
                print("|", end=" ")
                if board[i, j] == 0:
                    print(" ", end=" ")
                elif board[i, j] == 1:
                    if move_history[-1] == (i, j):
                        print(
                            bcolors.GREEN + self.player1_symbol + bcolors.ENDC, end=" "
                        )
                    else:
                        print(self.player1_symbol, end=" ")
                else:
                    if move_history[-1] == (i, j):
                        print(
                            bcolors.GREEN + self.player2_symbol + bcolors.ENDC, end=" "
                        )
                    else:
                        print(self.player2_symbol, end=" ")
            print("|")
        print(" ---" * self.column_number)
