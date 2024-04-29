from .config import load_config
from .alpha_vit import AlphaVit
from .game import TicTacToe
from .modes import Mode
import logging
import ray
import sys


class App:

    def __init__(self, config_path: str, mode: Mode) -> None:
        self.config = load_config(config_path)
        self.mode = mode

        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        """Run the app with the given mode."""
        if self.mode == Mode.TRAINING:
            self.train()
        elif self.mode == Mode.EVALUATE:
            self.evaluate()
        else:
            self.logger.error(f"Invalid mode {self.mode}")
            sys.exit(1)

    def train(self) -> None:
        """Train the model with the given config."""
        self.logger.info("Starting training")
        ray.init(num_cpus=10)

        alpha_vit = AlphaVit(self.config, TicTacToe(3, 3))
        alpha_vit.learn()

        ray.shutdown()

    def evaluate(self) -> None:
        """Evaluate the model with the given config."""
        pass
