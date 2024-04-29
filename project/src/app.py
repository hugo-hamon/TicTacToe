from .config import Config


class App:

    def __init__(self, config: Config) -> None:
        self.config = config

    def run(self) -> None:
        """Run tic-tac-toe game based on the configuration."""
        pass
