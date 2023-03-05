"""
This file is where your agent's logic is kept. Define a bidding policy, factory placement policy, as well as a policy for playing the normal phase of the game

The tutorial will learn an RL agent to play the normal phase and use heuristics for the other two phases.

Note that like the other kits, you can only debug print to standard error e.g. print("message", file=sys.stderr)
"""

import os.path as osp
import sys
import numpy as np
import torch as th
from stable_baselines3.ppo import PPO
from lux.config import EnvConfig
from wrappers import SimpleUnitDiscreteController, SimpleUnitObservationWrapper
from luxai_s2.state import ObservationStateDict

# change this to use weights stored elsewhere
# make sure the model weights are submitted with the other code files
# any files in the logs folder are not necessary. Make sure to exclude the .zip extension here
MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"

def Valid(i, j):
    return i >= 0 and j >= 0 and i < 48 and j < 48


def AdjToFactory(x, y, i, j):
    dist_x = abs(x - i)
    dist_y = abs(y - j)
    if dist_x == 2 and dist_y == 2: return False
    if dist_x == 2 or dist_y == 2: return True
    return False

def place_factory(player, obs: ObservationStateDict):
    my_obs = obs["teams"][player]
    board = obs["board"]

    # print(step, self.player, file=sys.stderr)
    # print("hello", self.env_cfg, file=sys.stderr)

    if my_obs["metal"] == 0:
        return dict()

    potential_spawns = list(zip(*np.where(board["valid_spawns_mask"] == 1)))

    pos = potential_spawns[np.random.randint(0, len(potential_spawns))]

    lowest_rubble = 101

    # Find location that is adjacent to ice with minimal rubble available
    for spawn_loc in potential_spawns:
        x, y = spawn_loc
        min_rubble = 100
        count_zeros = 0
        has_ice = False
        for i in range(x - 2, x + 3):
            for j in range(y - 2, y + 2):
                if not Valid(i, j):
                    continue
                if AdjToFactory(x, y, i, j):
                    if board["ice"][i, j] == 1:
                        has_ice = True
                    min_rubble = min(min_rubble, board["rubble"][i, j])
                    if board["rubble"][i, j] == 0:
                        count_zeros += 1
        min_rubble -= count_zeros  # count_zeros is nonzero only if min_rubble is 0.
        if has_ice and min_rubble < lowest_rubble:
            lowest_rubble = min_rubble
            pos = spawn_loc

    metal = min(150, my_obs["metal"])
    water = min(150, my_obs["water"])
    return dict(spawn=pos, metal=metal, water=water)


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = env_cfg

        directory = osp.dirname(__file__)
        self.policy = PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))

        self.controller = SimpleUnitDiscreteController(self.env_cfg)

    def bid_policy(self, step: int, obs, remainingOverageTime: int = 60):
        # the policy here is the same one used in the RL tutorial: https://www.kaggle.com/code/stonet2000/rl-with-lux-2-rl-problem-solving
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self, step: int, obs, remainingOverageTime: int = 60):
        """
        This policy will place a single factory with all the starting resources
        near a random ice tile
        """
        my_obs = obs["teams"][self.player]
        board = obs["board"]

        # print(step, self.player, file=sys.stderr)
        # print("hello", self.env_cfg, file=sys.stderr)

        if my_obs["metal"] == 0:
            return dict()

        potential_spawns = list(zip(*np.where(board["valid_spawns_mask"] == 1)))

        pos = potential_spawns[np.random.randint(0, len(potential_spawns))]

        lowest_rubble = 101

        # Find location that is adjacent to ice with minimal rubble available
        for spawn_loc in potential_spawns:
            x, y = spawn_loc
            min_rubble = 100
            count_zeros = 0
            has_ice = False
            for i in range(x - 2, x + 3):
                for j in range(y - 2, y + 2):
                    if not Valid(i, j):
                        continue
                    if AdjToFactory(x, y, i, j):
                        if board["ice"][i, j] == 1:
                            has_ice = True
                        min_rubble = min(min_rubble, board["rubble"][i, j])
                        if board["rubble"][i, j] == 0:
                            count_zeros += 1
            min_rubble -= count_zeros  # count_zeros is nonzero only if min_rubble is 0.
            if has_ice and min_rubble < lowest_rubble:
                lowest_rubble = min_rubble
                pos = spawn_loc

        metal = min(self.env_cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["metal"])
        water = min(self.env_cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["water"])
        return dict(spawn=pos, metal=metal, water=water)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        # first convert observations using the same observation wrapper you used for training
        # note that SimpleUnitObservationWrapper takes input as the full observation for both players and returns an obs for players
        raw_obs = dict(player_0=obs, player_1=obs)
        obs = SimpleUnitObservationWrapper.convert_obs(raw_obs, env_cfg=self.env_cfg)
        obs = obs[self.player]

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks(self.player, raw_obs))
                .unsqueeze(0)
                .bool()
            )
            
            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.policy.policy.features_extractor(obs.unsqueeze(0))
            x = self.policy.policy.mlp_extractor.shared_net(features)
            logits = self.policy.policy.action_net(x) # shape (1, N) where N=12 for the default controller

            logits[~action_mask] = -1e8 # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy() # shape (1, 1)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(
            self.player, raw_obs, actions[0]
        )

        # commented code below adds watering lichen which can easily improve your agent
        shared_obs = raw_obs[self.player]
        factories = shared_obs["factories"][self.player]
        for unit_id in factories.keys():
            factory = factories[unit_id]
            if 1000 - step < 201 and factory["cargo"]["water"] > 200:
                lux_action[unit_id] = 2 # water and grow lichen at the very end of the game

        return lux_action
