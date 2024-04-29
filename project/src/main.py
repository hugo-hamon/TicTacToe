from algorithms import get_best_move, Algorithm
from tictactoe import TicTacToe, Player
from time import time

DEPTH = 4

if __name__ == "__main__":
    game = TicTacToe(6, 6, 4)

    while not game.is_over():
        game.display_game()
        # if game.get_current_player() == Player.PLAYER1:
        #     row, column = map(int, input("Enter row and column: ").split())
        #     game.play(row, column)
        # else:
        start = time()
        best_move = get_best_move(game, DEPTH, game.get_current_player(), Algorithm.ALPHABETA)
        print(f"Time: {time() - start:.2f}s")
        if best_move is None:
            break
        game.play(best_move[0], best_move[1])

    game.display_game()
    