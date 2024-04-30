from __future__ import annotations
from ..game.tictactoe import TicTacToe
from typing import Optional
import numpy as np
import random


class Node:

    def __init__(self, game: TicTacToe, parent=None, action_taken=None) -> None:
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
        return q_value + np.sqrt(2 * np.log(self.visit_count) / (child.visit_count + 1))

    def expand(self) -> None:
        """Expand the node by adding a child."""
        for move in self.game.get_possible_moves():
            game_copy = self.game.copy()
            game_copy.play(move[0], move[1])
            child = Node(game_copy, self, move)
            self.children.append(child)

    def backpropagate(self, value: float) -> None:
        """Backpropagate the value to the root node."""
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:

    def __init__(self, game: TicTacToe) -> None:
        self.game = game
        self.player = game.get_current_player()

    def search(self, simulations: int) -> tuple[int, int]:
        """Return the best move after running the MCTS algorithm."""
        root = Node(self.game.copy())
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

        return self.get_best_move(root)

    def simulate(self, game: TicTacToe) -> int:
        """Simulate a game from the current state."""
        return game.line_open(self.player)

        while not game.is_over():
            move = random.choice(game.get_possible_moves())
            game.play(move[0], move[1])

        return game.get_value(self.player)

    def get_best_move(self, root: Node) -> tuple[int, int]:
        """Return the best move from the root node."""
        best_move = None
        best_value = -np.inf

        for child in root.children:
            if child.visit_count > best_value:
                best_value = child.visit_count
                best_move = child.action_taken

        return best_move


if __name__ == "__main__":
    game = TicTacToe(3, 3, 3)
    mcts = MCTS(game)
    move = mcts.search(1000)
    print(move)