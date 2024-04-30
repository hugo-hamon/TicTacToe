from .manager import Manager
from ..game.game import Game
from ..config import Config
from typing import Optional
import random


class RandomManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def reset(self) -> None:
        """Reset the manager"""
        pass

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return a random move"""
        possible_moves = game.get_possible_moves()
        if not possible_moves:
            return None
        return random.choice(possible_moves)
