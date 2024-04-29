from __future__ import annotations
from TicTacToeRL.project.src.game.tictactoe import TicTacToe, Player, get_opponent
from typing import Optional
from enum import Enum
import math


class Algorithm(Enum):
    MINIMAX = 1
    ALPHABETA = 2
    NEGABETA = 3


def get_score(game: TicTacToe, player: Player) -> float:
    """Return the score of the game."""
    winner = game.get_winner()
    if winner is None:
        return game.line_open(player) - game.line_open(get_opponent(player))
    elif winner == player:
        return game.row_number * game.column_number
    return -game.row_number * game.column_number


def alphabeta(game: TicTacToe, depth: int, alpha: float, beta: float, maximizing: bool, player: Player, count: int) -> tuple[int, float, Optional[tuple[int, int]]]:
    """Return the best move and its score."""
    if depth == 0 or game.is_over():
        return 1, get_score(game, player), None

    best_move = None
    total_count = 0
    if maximizing:
        value = float('-inf')
        for move in game.get_possible_moves():
            game_copy = game.copy()
            game_copy.play(move[0], move[1])
            new_count, move_value, _ = alphabeta(
                game_copy, depth - 1, alpha, beta, False, player, count
            )
            total_count += new_count
            if move_value > value:
                value = move_value
                best_move = move
            alpha = max(alpha, value)
            if value >= beta:
                break
    else:
        value = float('inf')
        for move in game.get_possible_moves():
            game_copy = game.copy()
            game_copy.play(move[0], move[1])
            new_count, move_value, _ = alphabeta(
                game_copy, depth - 1, alpha, beta, True, player, count
            )
            total_count += new_count
            if move_value < value:
                value = move_value
                best_move = move
            beta = min(beta, value)
            if value <= alpha:
                break

    return total_count, value, best_move


def minimax(game: TicTacToe, depth: int, maximizing: bool, player: Player) -> tuple[float, Optional[tuple[int, int]]]:
    """Return the best move and its score."""
    if depth == 0 or game.is_over():
        return get_score(game, player), None

    best_move = None

    if maximizing:
        value = float('-inf')
        for move in game.get_possible_moves():
            game_copy = game.copy()
            game_copy.play(move[0], move[1])
            move_value, _ = minimax(game_copy, depth - 1, False, player)
            if move_value > value:
                value = move_value
                best_move = move
    else:
        value = float('inf')
        for move in game.get_possible_moves():
            game_copy = game.copy()
            game_copy.play(move[0], move[1])
            move_value, _ = minimax(game_copy, depth - 1, True, player)
            if move_value < value:
                value = move_value
                best_move = move

    return value, best_move


def negabeta(game: TicTacToe, depth: int, alpha: float, beta: float, color: int, player: Player) -> tuple[float, Optional[tuple[int, int]]]:
    """Return the best move and its score using negabeta."""
    if depth == 0 or game.is_over():
        return color * get_score(game, player), None

    best_move = None
    value = float('-inf')

    for move in game.get_possible_moves():
        game_copy = game.copy()
        game_copy.play(move[0], move[1])
        move_value, _ = negabeta(
            game_copy, depth - 1, -beta, -alpha, -color, player
        )
        move_value = -move_value
        if move_value > value:
            value = move_value
            best_move = move
        alpha = max(alpha, value)
        if value >= beta:
            break

    return value, best_move


def get_best_move(game: TicTacToe, depth: int, player: Player, algorithm: Algorithm) -> tuple[int, int]:
    """Return the best move for the given player."""
    move = None
    if algorithm == Algorithm.ALPHABETA:
        count, score, move = alphabeta(
            game, depth, float('-inf'), float('inf'), True, player, 0
        )
    if algorithm == Algorithm.MINIMAX:
        score, move = minimax(game, depth, True, player)
    if algorithm == Algorithm.NEGABETA:
        score, move = negabeta(
            game, depth, float('-inf'), float('inf'), 1, player
        )
    if move is None:
        raise Exception("No move found")
    return move
