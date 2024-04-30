from ..manager.alpha_zero_manager import AlphaZeroManager
from ..manager.alpha_beta_manager import AlphaBetaManager
from ..manager.nega_max_manager import NegaMaxManager
from ..manager.minimax_manager import MinimaxManager
from ..manager.random_manager import RandomManager
from ..manager.mcts_manager import MCTSManager
from ..manager.user_manager import UserManager
from ..manager.manager import Manager
from ..game.game import Game
from ..config import Config
import logging
import sys


def match_manager(config: Config, manager_name: str, depth: int) -> Manager:
    """Return a manager from the config"""
    match manager_name:
        case "human":
            return UserManager(config)
        case "mcts":
            return MCTSManager(config)
        case "random":
            return RandomManager(config)
        case "alpha_zero":
            return AlphaZeroManager(config)
        case "minimax":
            return MinimaxManager(config, depth)
        case "nega_max":
            return NegaMaxManager(config, depth)
        case "alpha_beta":
            return AlphaBetaManager(config, depth)
        case _:
            logging.error(
                f'Found "{manager_name}" in class App in method match_manager'
            )
            sys.exit(1)


def get_user_move(game: Game) -> tuple[int, int]:
    """Get a move from the user"""
    while True:
        try:
            row, column = input("Enter your move (row column): ").split()
            row, column = int(row), int(column)
            if game.is_valid_move(row, column):
                return row, column
            print("Invalid move. Please try again.")
        except ValueError:
            print("Invalid move. Please try again.")
        except KeyboardInterrupt:
            print("\nExiting the game.")
            sys.exit(0)
        except EOFError:
            print("\nExiting the game.")
            sys.exit(0)
        except Exception as e:
            print(f"An error occurred: {e}. Please try again.")
