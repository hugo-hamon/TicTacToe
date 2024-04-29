from dataclasses import dataclass
from dacite.core import from_dict
import toml

"""

[user]
player1_algorithm = "alpha_beta"
player2_algorithm = "human"
player1_depth = 3
player2_depth = 3

[graphics]
graphics_enabled = true
player1_symbol = "X"
player2_symbol = "O"

[game]
board_width = 5
board_height = 5
to_align = 4

[mcts]
exploration_constant = 2.0
simulations_per_iteration = 100

[model]
iterations = 20
self_play_iterations = 100
epochs = 4
batch_size = 64
temperature = 1

"""


@dataclass
class UserConfig:
    player1_algorithm: str
    player2_algorithm: str
    player1_depth: int
    player2_depth: int


@dataclass
class GraphicsConfig:
    graphics_enabled: bool
    player1_symbol: str
    player2_symbol: str


@dataclass
class GameConfig:
    board_width: int
    board_height: int
    to_align: int


@dataclass
class MCTSConfig:
    exploration_constant: float
    simulations_per_iteration: int


@dataclass
class ModelConfig:
    iterations: int
    self_play_iterations: int
    epochs: int
    batch_size: int
    temperature: float


@dataclass
class Config:
    user: UserConfig
    graphics: GraphicsConfig
    game: GameConfig
    mcts: MCTSConfig
    model: ModelConfig


def load_config(config_path: str) -> Config:
    """Load the config from a file."""
    return from_dict(data_class=Config, data=toml.load(config_path))
