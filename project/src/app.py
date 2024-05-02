from .utils.manager_func import match_manager, get_user_move
from .graphics.shell.shell_graphics import ShellGraphics
from .manager.user_manager import UserManager
from .trainer.alpha_zero import AlphaZero
from .manager.manager import Manager
from .trainer.model import ResNet
from .config import load_config
from .game.game import Player
from typing import Optional
from .game.game import Game
import logging
import time
import eel
import ray
import os


class App:

    def __init__(self, config_path: str) -> None:
        self.config = load_config(config_path)

        self.player1: Optional[Manager] = None
        self.player2: Optional[Manager] = None

        self.game = None
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Run tic-tac-toe game based on the configuration."""

        if self.config.alpha_zero.training_enabled:
            self.train()

        self.player1 = match_manager(
            self.config,
            self.config.user.player1_algorithm,
            self.config.user.player1_depth,
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
        self.game.init_open_lines()

        if self.config.graphics.graphics_enabled:
            if self.config.graphics.shell_graphics:
                graphics = ShellGraphics(self.config)
                self.run_with_shell_graphics(graphics)
            else:
                eel.init("src/graphics/web")
                self.expose_functions()
                eel.start(
                    "index.html",
                    mode="firefox",
                    cmdline_args=["--start-fullscreen"],
                    shutdown_delay=5
                )

    # AlphaZero functions
    def train(self) -> None:
        """Train the model."""
        if self.config.alpha_zero.using_ray:
            ray.init(num_cpus=10)

        model = ResNet(self.config, 3, 4, 64)
        model = AlphaZero(self.config, model)
        model.learn()

        if self.config.alpha_zero.using_ray:
            ray.shutdown()
        self.logger.info("Training completed")

    # Shell functions
    def run_with_shell_graphics(self, graphics: ShellGraphics) -> None:
        """Run the game with shell graphics."""
        if not self.game:
            raise ValueError("Game not initialized")
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
        if not self.game:
            raise ValueError("Game not initialized")
        if player == Player.PLAYER1 and isinstance(self.player1, UserManager):
            move = get_user_move(self.game)
            self.player1.set_move(move)
        elif player == Player.PLAYER2 and isinstance(self.player2, UserManager):
            move = get_user_move(self.game)
            self.player2.set_move(move)

    # eel functions
    def expose_functions(self) -> None:
        """Expose functions to JavaScript"""
        functions = self.__dir__()
        for function in functions:
            if function.startswith("eel_"):
                eel.expose(getattr(self, function))

    def eel_stop(self) -> None:
        """Stop the program."""
        self.logger.info("Stopping the program")
        os._exit(0)

    def eel_get_grid_size(self) -> tuple[int, int]:
        """Get the grid size."""
        return self.config.game.row_number, self.config.game.column_number

    def eel_get_board(self) -> Optional[list[list[str]]]:
        """Get the board."""
        if self.game:
            return self.game.get_board().tolist()

    def eel_get_current_player(self) -> Optional[int]:
        """Get the current player."""
        if self.game:
            return self.game.get_current_player().value

    def eel_make_move(self, row: int, column: int) -> int:
        """Make a move."""
        return_code = -1
        if self.game:
            if not self.game.is_valid_move(row, column):
                return return_code

            current_player = self.game.get_current_player()
            if isinstance(self.player1, UserManager) and current_player == Player.PLAYER1:
                self.player1.set_move((row, column))
                return_code = 0
            elif isinstance(self.player2, UserManager) and current_player == Player.PLAYER2:
                self.player2.set_move((row, column))
                return_code = 0

        return return_code

    def eel_get_winner(self) -> Optional[int]:
        """Get the winner."""
        if self.game:
            winner = self.game.get_winner()
            if winner:
                return winner.value

    def eel_update_game(self) -> None:
        """Update the game."""
        if self.game:
            self.game.update()

    def eel_is_current_player_human(self) -> bool:
        """Check if the current player is human."""
        if self.game:
            current_player = self.game.get_current_player()
            if current_player == Player.PLAYER1 and isinstance(self.player1, UserManager):
                return True
            elif current_player == Player.PLAYER2 and isinstance(self.player2, UserManager):
                return True
        return False

    def eel_is_game_over(self) -> Optional[bool]:
        """Check if the game is over."""
        if self.game:
            return self.game.is_over()

    def eel_reset_game(self) -> None:
        """Reset the game."""
        if self.game:
            self.game.reset()
            self.game.init_open_lines()
            self.logger.info("Game reset")

    def eel_undo(self) -> None:
        """Go to the last human move."""
        if self.game:
            self.game.undo()
            is_current_player_human = self.eel_is_current_player_human()
            if not is_current_player_human:
                self.game.undo()
