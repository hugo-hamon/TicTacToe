from torch.utils.tensorboard.writer import SummaryWriter
import torch.nn.functional as F
from .alpha_mcts import MCTS
from ..game.game import Game
from ..config import Config
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import logging
import random
import torch
import time
import ray


class AlphaZero:

    def __init__(self, config: Config, model: nn.Module) -> None:
        self.config = config
        self.model = model
        self.logger = logging.getLogger(__name__)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.alpha_zero.learning_rate
        )
        self.writer = SummaryWriter(log_dir="log/scalars/")
        self.mcts = MCTS(self.config, self.model)

    def train(self, memory: list) -> tuple:
        """Train the model using the memory."""
        random.shuffle(memory)
        policy_losses = []
        value_losses = []
        for batchIdx in range(0, len(memory), self.config.alpha_zero.batch_size):
            sample = memory[
                batchIdx: min(
                    len(memory) - 1, batchIdx +
                    self.config.alpha_zero.batch_size
                )
            ]
            if len(sample) == 0:
                break
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = (
                np.array(state),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            state = torch.tensor(state, dtype=torch.float32)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)

            out_policy, out_value = self.model(state)

            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return np.mean(policy_losses), np.mean(value_losses)

    def self_play(self) -> list:
        memory = []
        player = 1
        game = Game(self.config, {})
        action_size = self.config.game.row_number * self.config.game.column_number
        while True:
            neutral_game = game.change_perspective(player)
            action_probs = self.mcts.search(neutral_game)

            memory.append((neutral_game.copy(), action_probs, player))

            action = np.random.choice(action_size, p=action_probs)

            row = action // self.config.game.column_number
            column = action % self.config.game.column_number
            game.play(row, column)

            value, is_terminal = game.get_value_and_terminated()

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else -value
                    returnMemory.append(
                        (
                            hist_neutral_state.get_encoded_state(),
                            hist_action_probs,
                            hist_outcome,
                        )
                    )
                return returnMemory

            player = -player

    def learn(self) -> None:
        """Learn the model by self-playing."""
        for iteration in range(self.config.alpha_zero.training_iterations):
            memory = []

            self.model.eval()
            start = time.time()
            if not self.config.alpha_zero.using_ray:
                for _ in tqdm(range(self.config.alpha_zero.self_play_iterations)):
                    memory.extend(self.self_play())
            else:
                results_ids = [
                    self_play.remote(self.config, self.mcts) for _ in range(
                        self.config.alpha_zero.self_play_iterations
                    )
                ]
                results = ray.get(results_ids)
                for result in results:
                    memory.extend(result)
            end = time.time()
            self.logger.info(
                f"Self-play iterations took {end - start:.2f}s"
            )

            self.model.train()
            for _ in range(self.config.alpha_zero.epochs):
                policy_loss, value_loss = self.train(memory)
                self.writer.add_scalar("Policy Loss", policy_loss, iteration)
                self.writer.add_scalar("Value Loss", value_loss, iteration)

        # Save the model
        row_number = self.config.game.row_number
        column_number = self.config.game.column_number
        training_iterations = self.config.alpha_zero.training_iterations
        to_align = self.config.game.to_align
        torch.save(
            self.model, f"model/{row_number}x{column_number}x{to_align}_{training_iterations}.pt")
        self.writer.close()
        self.logger.info("Model saved")


@ray.remote
def self_play(config: Config, mcts: MCTS) -> list:
    memory = []
    player = 1
    game = Game(config, {})
    action_size = config.game.row_number * config.game.column_number
    while True:
        neutral_game = game.change_perspective(player)
        action_probs = mcts.search(neutral_game)

        memory.append((neutral_game.copy(), action_probs, player))

        action = np.random.choice(action_size, p=action_probs)

        row = action // config.game.column_number
        column = action % config.game.column_number
        game.play(row, column)

        value, is_terminal = game.get_value_and_terminated()

        if is_terminal:
            returnMemory = []
            for hist_neutral_state, hist_action_probs, hist_player in memory:
                hist_outcome = value if hist_player == player else -value
                returnMemory.append(
                    (
                        hist_neutral_state.get_encoded_state(),
                        hist_action_probs,
                        hist_outcome,
                    )
                )
            return returnMemory

        player = -player
