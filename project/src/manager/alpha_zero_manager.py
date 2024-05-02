from ..trainer.alpha_mcts import MCTS
from ..trainer.model import ResNet
from .manager import Manager
from ..game.game import Game
from ..config import Config
from typing import Optional
import numpy as np
import logging
import torch
import os


class AlphaZeroManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.model = ResNet(config, 3, 4, 64)
        self.logger = logging.getLogger(__name__)

        # load the model
        row_number = config.game.row_number
        column_number = config.game.column_number
        to_align = config.game.to_align

        for file in os.listdir("model"):
            if file.startswith(f"{row_number}x{column_number}x{to_align}") and file.endswith(".pt"):
                self.model = torch.load(f"model/{file}")
                self.logger.info(f"Model loaded from model/{file}")
                break
        

    def reset(self) -> None:
        """Reset the manager"""
        pass

    @torch.no_grad()
    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the best move using the AlphaZero algorithm"""
        mcts = MCTS(self.config, self.model)
        game = game.change_perspective(game.current_player.value)
        action_probs = mcts.search(game)
        best_action = int(np.argmax(action_probs))
        row = best_action // self.config.game.column_number
        column = best_action % self.config.game.column_number
        return row, column
