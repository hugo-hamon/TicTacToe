from src.app import App
import argparse
import logging

LOGGING_CONFIG = {
    "level": logging.INFO,
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "datefmt": "%d-%b-%y %H:%M:%S",
    "filename": "log/log.log",
    "filemode": "w",
}

DEFAULT_CONFIG_PATH = "config/default.toml"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="TicTacToeRL",
        description="Play Tic-Tac-Toe against an AI using Reinforcement Learning.",
    )
    parser.add_argument(
        "--config",
        "-c",
        metavar="CONFIG_FILE",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the configuration file. If not provided, uses the default config.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logging.basicConfig(**LOGGING_CONFIG)
    logger.info("Starting run.py")

    args = parse_args()
    config_path = args.config

    logger.info(f"Using config file {config_path}")

    app = App(config_path)
    app.run()
