from dataclasses import dataclass
from dacite.core import from_dict
import toml


@dataclass
class UserConfig:
    player1_algorithm: str
    player2_algorithm: str
    player1_depth: int
    player2_depth: int


@dataclass
class GraphicsConfig:
    graphics_enabled: bool
    shell_graphics: bool
    player1_symbol: str
    player2_symbol: str


@dataclass
class GameConfig:
    row_number: int
    column_number: int
    to_align: int


@dataclass
class MCTSConfig:
    exploration_constant: float
    simulations_per_iteration: int


@dataclass
class ModelConfig:
    training_enabled: bool
    training_iterations: int
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
