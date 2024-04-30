from .manager import Manager
from ..game.game import Game
from ..config import Config
from typing import Optional


class MCTSManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def reset(self) -> None:
        """Reset the manager"""
        pass

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the best move using the MCTS algorithm"""
        pass
