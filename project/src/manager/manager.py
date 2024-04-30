from abc import ABC, abstractmethod
from ..game.game import Game
from typing import Optional
from ..config import Config


class Manager(ABC):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return a move for the given game"""
        return NotImplemented

    @abstractmethod
    def reset(self) -> None:
        """Reset the manager"""
        return NotImplemented
