from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import torch


class Game(ABC):

    def __init__(self) -> None:
        self.action_size = 0

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Return a representation of the initial state of the game."""
        pass

    @abstractmethod
    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        """Return the state that results from taking action in state."""
        pass

    @abstractmethod
    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        """Return a list of the allowable moves at this point."""
        pass

    @abstractmethod
    def get_value_and_terminated(
        self, state: np.ndarray, action: int
    ) -> Tuple[int, bool]:
        """Return the value and whether the game has terminated after taking action in state."""
        pass

    @abstractmethod
    def get_opponent(self, player: int) -> int:
        """Return the opponent of the given player."""
        pass

    @abstractmethod
    def get_opponent_value(self, value: int) -> int:
        """Return the opponent of the given value."""
        pass

    @abstractmethod
    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        """Return the state from the perspective of the opponent."""
        pass

    @abstractmethod
    def get_encoded_state(self, state: np.ndarray) -> np.ndarray:
        """Return a tensor representing the state."""
        pass


class TicTacToe(Game):

    def __init__(self, size: int, to_align: int) -> None:
        self.size = size
        self.to_align = to_align
        self.action_size = size * size

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.size, self.size), dtype=int)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        row, column = divmod(action, self.size)
        new_state = state.copy()
        new_state[row, column] = player
        return new_state

    def get_valid_moves(self, state: np.ndarray) -> np.ndarray:
        return (state.reshape(-1) == 0).astype(int)

    def check_win(self, state: np.ndarray, action: int) -> bool:
        row, column = divmod(action, self.size)
        player = state[row, column]

        # Check row
        for row in state:
            for i in range(self.size - self.to_align + 1):
                if row[i] == player and all(
                    row[j] == player for j in range(i + 1, i + self.to_align)
                ):
                    return True

        # Check column
        for col in state.T:
            for i in range(self.size - self.to_align + 1):
                if col[i] == player and all(
                    col[j] == player for j in range(i + 1, i + self.to_align)
                ):
                    return True

        # Check diagonals
        for i in range(self.size - self.to_align + 1):
            for j in range(self.size - self.to_align + 1):
                if state[i, j] == player and all(
                    state[i + k, j + k] == player for k in range(1, self.to_align)
                ):
                    return True

                if state[i, j + self.to_align - 1] == player and all(
                    state[i + k, j + self.to_align - 1 - k] == player
                    for k in range(1, self.to_align)
                ):
                    return True

        return False

    def get_value_and_terminated(
        self, state: np.ndarray, action: int
    ) -> Tuple[int, bool]:
        if action != None and self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int) -> int:
        return -player

    def get_opponent_value(self, value: int) -> int:
        return -value

    def change_perspective(self, state: np.ndarray, player: int) -> np.ndarray:
        return state * player

    def get_encoded_state(self, state: np.ndarray) -> torch.Tensor:
        image = torch.tensor(self.board_to_flat_image(state, 1), dtype=torch.float32)[
            None, None, :, :
        ]
        return image / 255.0

    def board_to_flat_image(self, state: np.ndarray, image_size: int) -> np.ndarray:
        """Convert the board to a flat image."""
        image = np.zeros((image_size, image_size * self.size * self.size), dtype=int)
        for i in range(self.size):
            for j in range(self.size):
                if state[i, j] == 1:
                    image[
                        :,
                        i * self.size * image_size
                        + j * image_size : i * self.size * image_size
                        + (j + 1) * image_size,
                    ] = 128
                elif state[i, j] == -1:
                    image[
                        :,
                        i * self.size * image_size
                        + j * image_size : i * self.size * image_size
                        + (j + 1) * image_size,
                    ] = 255

        return image

