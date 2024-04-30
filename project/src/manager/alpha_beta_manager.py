from ..game.game import Game, Player
from .manager import Manager
from ..config import Config
from typing import Optional


class AlphaBetaManager(Manager):

    def __init__(self, config: Config, depth: int) -> None:
        super().__init__(config)
        self.depth = depth
        if depth < 1:
            raise ValueError("AlphaBeta depth should be at least 1")

    def reset(self) -> None:
        """Reset the manager"""
        pass

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the best move using the alpha beta algorithm"""
        _, best_move = self.alphabeta(
            game,
            self.depth,
            float("-inf"),
            float("inf"),
            True,
            game.get_current_player(),
        )
        return best_move

    def alphabeta(
        self,
        game: Game,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        player: Player,
    ) -> tuple[float, Optional[tuple[int, int]]]:
        """Return the best move and its score."""
        if depth == 0 or game.is_over():
            return self.get_score(game, player), None

        best_move = None
        if maximizing:
            value = float("-inf")
            for move in game.get_possible_moves():
                game_copy = game.copy()
                game_copy.play(move[0], move[1])

                move_value, _ = self.alphabeta(
                    game_copy, depth - 1, alpha, beta, False, player
                )
                if move_value > value:
                    value = move_value
                    best_move = move
                alpha = max(alpha, value)
                if value >= beta:
                    break
        else:
            value = float("inf")
            for move in game.get_possible_moves():
                game_copy = game.copy()
                game_copy.play(move[0], move[1])
                move_value, _ = self.alphabeta(
                    game_copy, depth - 1, alpha, beta, True, player
                )
                if move_value < value:
                    value = move_value
                    best_move = move
                beta = min(beta, value)
                if value <= alpha:
                    break

        return value, best_move

    def get_score(self, game: Game, player: Player) -> float:
        """Return the score of the game."""
        move_history = game.get_move_history()
        if len(move_history) < self.config.game.to_align * 2 - 1:
            winner = None
        else:
            winner = game.get_winner()
        if winner is None:
            return game.line_open(player) - game.line_open(game.get_opponent(player))
        elif winner == player:
            return game.row_number * game.column_number
        return -game.row_number * game.column_number

