from .utils.manager_func import match_manager, get_user_move
from .graphics.shell.shell_graphics import ShellGraphics
from .manager.user_manager import UserManager
from .manager.manager import Manager
from .config import load_config
from .game.game import Player
from typing import Optional
from .game.game import Game
import logging
import time


class App:

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

        self.player1: Optional[Manager] = None
        self.player2: Optional[Manager] = None

        self.game = None
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Run tic-tac-toe game based on the configuration."""
        self.player1 = match_manager(
            self.config,
            self.config.user.player1_algorithm,
            self.config.user.player2_depth,
        )

        self.player2 = match_manager(
            self.config,
            self.config.user.player2_algorithm,
            self.config.user.player2_depth,
        )
        self.logger.info(
            f"Player 1 is using {self.config.user.player1_algorithm}"
            + f" and Player 2 is using {self.config.user.player2_algorithm}"
        )

        self.game = Game(
            self.config,
            {
                Player.PLAYER1.name: self.player1.get_move,
                Player.PLAYER2.name: self.player2.get_move,
            },
        )

        if self.config.graphics.graphics_enabled:
            if self.config.graphics.shell_graphics:
                graphics = ShellGraphics(self.config)
                self.run_with_shell_graphics(graphics)

    def run_with_shell_graphics(self, graphics: ShellGraphics) -> None:
        """Run the game with shell graphics."""
        self.logger.info("Running the game with shell graphics")
        graphics.draw(self.game)
        last_state = self.game.get_board().copy()
        while not self.game.is_over():
            current_player = self.game.get_current_player()
            self.check_for_human_input(current_player)
            t1 = time.time()
            self.game.update()
            print(f"Time taken: {time.time() - t1:.2f}s")
            if not (last_state == self.game.get_board()).all():
                graphics.draw(self.game)
                last_state = self.game.get_board().copy()
        print("Game over")
        winner = self.game.get_winner()
        if winner:
            print(f"Player {winner.value} wins")
        else:
            print("It's a draw")

    def check_for_human_input(self, player: Player) -> None:
        """Check for human input."""
        if player == Player.PLAYER1 and isinstance(self.player1, UserManager):
            move = get_user_move(self.game)
            self.player1.set_move(move)
        elif player == Player.PLAYER2 and isinstance(self.player2, UserManager):
            move = get_user_move(self.game)
            self.player2.set_move(move)
