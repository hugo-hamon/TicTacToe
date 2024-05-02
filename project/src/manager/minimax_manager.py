from ..game.game import Game, Player
from .manager import Manager
from ..config import Config
from typing import Optional


class MinimaxManager(Manager):

    def __init__(self, config: Config, depth: int) -> None:
        super().__init__(config)
        self.depth = depth
        if depth < 1:
            raise ValueError("MiniMax depth should be at least 1")

    def reset(self) -> None:
        """Reset the manager"""
        pass

    def get_move(self, game: Game) -> Optional[tuple[int, int]]:
        """Return the best move using the minimax algorithm"""
        score, best_move = self.minimax(
            game, self.depth, True, game.get_current_player()
        )
        if score == -game.row_number * game.column_number:
            # If the score is the worst possible, return a move with minimax depth == 1
            _, best_move = self.minimax(
                game, 1, True, game.get_current_player()
            )
        return best_move

    def minimax(self, game: Game, depth: int, maximizing: bool, player: Player) -> tuple[float, Optional[tuple[int, int]]]:
        """Return the best move and its score."""
        if depth == 0 or game.is_over():
            return self.get_score(game, player), None

        best_move = None

        if maximizing:
            value = float('-inf')
            for move in game.get_possible_moves():
                game_copy = game.copy()
                game_copy.play(move[0], move[1])
                move_value, _ = self.minimax(
                    game_copy, depth - 1, False, player)
                if move_value > value:
                    value = move_value
                    best_move = move
        else:
            value = float('inf')
            for move in game.get_possible_moves():
                game_copy = game.copy()
                game_copy.play(move[0], move[1])
                move_value, _ = self.minimax(
                    game_copy, depth - 1, True, player)
                if move_value < value:
                    value = move_value
                    best_move = move

        return value, best_move

    def get_score(self, game: Game, player: Player) -> float:
        """Return the score of the game."""
        move_history = game.get_move_history()
        if len(move_history) < self.config.game.to_align * 2 - 1:
            winner = None
        else:
            winner = game.get_winner()
        if winner is None:
            return game.get_number_of_open_lines(player) - game.get_number_of_open_lines(
                game.get_opponent(player)
            )
        elif winner == player:
            return game.row_number * game.column_number
        return -game.row_number * game.column_number
