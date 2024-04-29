from torch.utils.tensorboard.writer import SummaryWriter
from .model import AlphaVitModel
import torch.nn.functional as F
from .config import Config
from tqdm import trange
from .mcts import MCTS
from .game import Game
from tqdm import tqdm
import numpy as np
import logging
import random
import torch
import time
import ray

LEARNING_RATE = 1e-3


class AlphaVit:

    def __init__(self, config: Config, game: Game) -> None:
        self.game = game
        self.config = config

        self.model = AlphaVitModel(
            input_size=(1, 9), path_size=1, latent_size=120, output_size=9
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.writer = SummaryWriter()

        self.mcts = MCTS(game, config, self.model)

    def train(self, memory) -> tuple:
        random.shuffle(memory)
        policy_losses = []
        value_losses = []
        for batchIdx in range(0, len(memory), self.config.model.batch_size):
            sample = memory[
                batchIdx : min(len(memory) - 1, batchIdx + self.config.model.batch_size)
            ]
            state, policy_targets, value_targets = zip(*sample)

            state, policy_targets, value_targets = (
                torch.stack(state).squeeze(1),
                np.array(policy_targets),
                np.array(value_targets).reshape(-1, 1),
            )

            policy_targets = torch.tensor(policy_targets, dtype=torch.float32)
            value_targets = torch.tensor(value_targets, dtype=torch.float32)
            out_policy, out_value = self.model(state)
            out_policy = out_policy.squeeze(0).squeeze(0)
            out_value = out_value.squeeze(0).squeeze(0)

            policy_loss = F.cross_entropy(out_policy, policy_targets.to(self.model.device))
            value_loss = F.mse_loss(out_value, value_targets.to(self.model.device))
            loss = policy_loss + value_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        return np.mean(policy_losses), np.mean(value_losses)

    def selfPlay(self) -> list:
        memory = []
        player = 1
        state = self.game.get_initial_state()

        while True:
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(neutral_state)

            memory.append((neutral_state, action_probs, player))

            action = np.random.choice(self.game.action_size, p=action_probs)

            state = self.game.get_next_state(state, action, player)

            value, is_terminal = self.game.get_value_and_terminated(state, action)

            if is_terminal:
                returnMemory = []
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = (
                        value
                        if hist_player == player
                        else self.game.get_opponent_value(value)
                    )
                    returnMemory.append(
                        (
                            self.game.get_encoded_state(hist_neutral_state),
                            hist_action_probs,
                            hist_outcome,
                        )
                    )
                return returnMemory

            player = self.game.get_opponent(player)

    def learn(self):
        for iteration in range(self.config.model.iterations):
            memory = []

            start_time = time.time()
            self.model.eval()
            """
            results_ids = [
                selfPlay.remote(self.game, self.mcts)
                for _ in range(self.config.model.self_play_iterations)
            ]
            results = ray.get(results_ids)
            for result in results:
                memory.extend(result)
            """

            for _ in tqdm(range(self.config.model.self_play_iterations)):
                memory.extend(self.selfPlay())
            

            logging.info(f"Self-play took {time.time() - start_time} seconds")

            self.model.train()
            mean_value_loss, mean_policy_loss = (0, 0)
            for _ in trange(self.config.model.epochs):
                policy_loss, value_loss = self.train(memory)
                mean_value_loss += value_loss
                mean_policy_loss += policy_loss
            mean_value_loss /= self.config.model.epochs
            mean_policy_loss /= self.config.model.epochs

            logging.info(
                f"Iteration {iteration}, value loss: {mean_value_loss}, policy loss: {mean_policy_loss}"
            )

            self.writer.add_scalar("value_loss", mean_value_loss, iteration)
            self.writer.add_scalar("policy_loss", mean_policy_loss, iteration)

            torch.save(self.model.state_dict(), f"model/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"model/optimizer_{iteration}.pt")


@ray.remote
def selfPlay(game: Game, mcts: MCTS) -> list:
    memory = []
    player = 1
    state = game.get_initial_state()

    while True:
        neutral_state = game.change_perspective(state, player)
        action_probs = mcts.search(neutral_state)

        memory.append((neutral_state, action_probs, player))

        action = np.random.choice(game.action_size, p=action_probs)

        state = game.get_next_state(state, action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)

        if is_terminal:
            returnMemory = []
            for hist_neutral_state, hist_action_probs, hist_player in memory:
                hist_outcome = (
                    value if hist_player == player else game.get_opponent_value(value)
                )
                returnMemory.append(
                    (
                        game.get_encoded_state(hist_neutral_state),
                        hist_action_probs,
                        hist_outcome,
                    )
                )
            return returnMemory

        player = game.get_opponent(player)
