from __future__ import annotations
from .manager import Manager
from ..game.game import Game
from ..config import Config
from typing import Optional
import numpy as np
import random


class MCTSManager(Manager):

    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def reset(self) -> None:
        """Reset the manager"""
        pass

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the best move using the MCTS algorithm"""
        mcts = MCTS(self.config, game)
        return mcts.search(self.config.mcts.simulations_per_iteration)


class Node:

    def __init__(self, config: Config, game: Game, parent=None, action_taken=None) -> None:
        self.config = config
        self.game = game
        self.parent = parent
        self.action_taken = action_taken

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
        q_value = 0 if child.visit_count == 0 else child.value_sum / child.visit_count
        return q_value + np.sqrt(np.log(self.visit_count) / (child.visit_count + 1)) * self.config.mcts.exploration_constant

    def expand(self) -> None:
        """Expand the node by adding a child."""
        for move in self.game.get_possible_moves():
            game_copy = self.game.copy()
            game_copy.play(move[0], move[1])
            child = Node(self.config, game_copy, self, move)
            self.children.append(child)

    def backpropagate(self, value: float) -> None:
        """Backpropagate the value to the root node."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:

    def __init__(self, config: Config, game: Game) -> None:
        self.config = config
        self.game = game
        self.player = game.get_current_player()

    def search(self, simulations: int) -> tuple[int, int]:
        """Return the best move after running the MCTS algorithm."""
        root = Node(self.config, self.game.copy())
        best_child = {"child": None, "visit_count": 0, "iteration": 0}

        for _ in range(simulations):
            node = root

            while node.is_fully_expanded():
                node = node.select()
                if node is None:
                    raise Exception("Node is None")

            is_terminal = node.game.is_over()
            value = node.game.get_value(node.game.get_current_player())
            value = -value

            if not is_terminal:
                value = self.simulate(node.game.copy())
                node.expand()

            node.backpropagate(value)

            if self.config.mcts.stable_iterations > 0:
                best_child["iteration"] += 1
                for child in root.children:
                    if child.visit_count > best_child["visit_count"]:
                        best_child["child"] = child
                        best_child["visit_count"] = child.visit_count
                        best_child["iteration"] = 0

                if best_child["iteration"] > self.config.mcts.stable_iterations:
                    break

        return self.get_best_move(root)

    def simulate(self, game: Game) -> int:
        """Simulate a game from the current state."""
        while not game.is_over():
            move = random.choice(game.get_possible_moves())
            game.play(move[0], move[1])
        return int(game.get_value(self.player))

    def get_best_move(self, root: Node) -> tuple[int, int]:
        """Return the best move from the root node."""
        best_move = None
        best_value = -np.inf
        for child in root.children:
            if child.visit_count > best_value:
                best_value = child.visit_count
                best_move = child.action_taken

        if best_move is None:
            raise Exception("Best move is None")

        return best_move
