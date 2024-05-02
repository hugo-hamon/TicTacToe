from __future__ import annotations
from ..game.game import Game
from ..config import Config
from typing import Optional
import torch.nn as nn
import numpy as np
import torch
import math


class Node:

    def __init__(
        self, config: Config, game: Game, parent=None, action_taken=None, prior=0
    ) -> None:
        self.config = config
        self.game = game
        self.parent = parent
        self.action_taken = action_taken
        self.prior = prior

        self.children: list[Node] = []

        self.visit_count = 0
        self.value_sum = 0

    def is_fully_expanded(self) -> bool:
        """Return whether all children have been expanded."""
        return len(self.children) > 0

    def select(self) -> Optional[Node]:
        """Return the child with the highest UCB score."""
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb

        return best_child

    def get_ucb(self, child: Node) -> float:
        """Return the UCB score of the child."""
        q_value = (
            0
            if child.visit_count == 0
            else 1 - ((child.value_sum / child.visit_count) + 1) / 2
        )
        return (
            q_value
            + self.config.mcts.exploration_constant
            * (math.sqrt(self.visit_count) / (child.visit_count + 1))
            * child.prior
        )

    def expand(self, policy: np.ndarray) -> None:
        """Expand the node by creating all possible children."""
        for action, prob in enumerate(policy):
            row = action // self.config.game.column_number
            column = action % self.config.game.column_number
            if prob > 0:
                game_copy = self.game.copy()
                game_copy.play(row, column)

                child = Node(self.config, game_copy, self, action, prob)
                self.children.append(child)

    def backpropagate(self, value: float) -> None:
        """Backpropagate the value to the root node."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:

    def __init__(self, config: Config, model: nn.Module) -> None:
        self.config = config
        self.model = model

    @torch.no_grad()
    def search(self, game: Game) -> np.ndarray:
        root = Node(self.config, game)

        for _ in range(self.config.mcts.simulations_per_iteration):
            node = root

            while node.is_fully_expanded():
                node = node.select()
                if node is None:
                    raise Exception("Node is None")

            value, is_terminal = node.game.get_value_and_terminated()
            value = -value

            if not is_terminal:
                # policy, value = self.model(node.game.get_encoded_state())
                policy, value = self.model(
                    torch.tensor(node.game.get_encoded_state()).unsqueeze(0)
                )
                policy = torch.softmax(policy, dim=1).squeeze(0).cpu().numpy()
                valid_moves = node.game.get_one_hot_valid_moves()
                policy *= valid_moves
                policy /= np.sum(policy)

                value = value.item()

                node.expand(policy)

            node.backpropagate(value)

        action_probs = np.zeros(self.config.game.row_number * self.config.game.column_number)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs