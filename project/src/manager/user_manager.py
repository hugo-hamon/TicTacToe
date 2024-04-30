from .manager import Manager
from ..game.game import Game
from ..config import Config
from typing import Optional


class UserManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.current_move = None

    def reset(self) -> None:
        """Reset the manager"""
        self.current_move = None

    def set_move(self, move: tuple[int, int]) -> None:
        """Set the move"""
        self.current_move = move

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return a move from the user"""
        move = self.current_move
        self.current_move = None
        return move
