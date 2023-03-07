import sys

import numpy as np
import torch as th

from lux.kit import process_action

from wrappers import LuxObservationWrapper, LuxController

from game_state import GameState

def Valid(i, j):
    return i >= 0 and j >= 0 and i < 48 and j < 48


def AdjToFactory(x, y, i, j):
    dist_x = abs(x - i)
    dist_y = abs(y - j)
    if dist_x == 2 and dist_y == 2: return False
    if dist_x == 2 or dist_y == 2: return True
    return False


class Agent:
    def __init__(self, ppo_model, state: GameState) -> None:
        np.random.seed(0)
        self.ppo_model = ppo_model
        self.state = state

        # TODO why ObsWrapper is not init here
        self.controller = LuxController(self.state)

    def bid_policy(self):
        return dict(faction="AlphaStrike", bid=0)

    def factory_placement_policy(self):
        """
        This policy will place a single factory with all the starting resources
        near a random ice tile and if possible close to an empty lot.
        """
        state = self.state
        my_obs = state.my_team

        if my_obs["metal"] == 0:
            return dict()

        # TODO process in Gamestate greedily
        potential_spawns = list(zip(*np.where(state.valid_spawns_mask)))
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
                        if state.ice[i, j] == 1:
                            has_ice = True
                        min_rubble = min(min_rubble, state.rubble[i, j])
                        if state.rubble[i, j] == 0:
                            count_zeros += 1
            min_rubble -= count_zeros  # count_zeros is nonzero only if min_rubble is 0.
            if has_ice and min_rubble < lowest_rubble:
                lowest_rubble = min_rubble
                pos = spawn_loc

        metal = min(state.cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["metal"])
        water = min(state.cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["water"])
        return dict(spawn=pos, metal=metal, water=water)

    def act(self):

        obs = LuxObservationWrapper.convert_obs(self.state)

        obs = th.from_numpy(obs).float()
        with th.no_grad():

            # to improve performance, we have a rule based action mask generator for the controller used
            # which will force the agent to generate actions that are valid only.
            action_mask = (
                th.from_numpy(self.controller.action_masks())
                .unsqueeze(0)
                .bool()
            )

            # SB3 doesn't support invalid action masking. So we do it ourselves here
            features = self.ppo_model.policy.features_extractor(obs.unsqueeze(0))
            x = self.ppo_model.policy.mlp_extractor.shared_net(features)
            logits = self.ppo_model.policy.action_net(x)  # shape (1, N) where N=12 for the default controller

            logits[~action_mask] = -1e8  # mask out invalid actions
            dist = th.distributions.Categorical(logits=logits)
            actions = dist.sample().cpu().numpy()  # shape (1, 1)

        # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        lux_action = self.controller.action_to_lux_action(actions[0])


        # grows lichen
        for unit_id in self.state.factories.keys():
            factory = self.state.factories[unit_id]
            if 1000 - self.state.step < 201 and factory["cargo"]["water"] > 200:
                lux_action[unit_id] = 2  # water and grow lichen at the very end of the game

        return lux_action

    def step(self):
        """
        Returns actions for one step of the environment.
        """
        if self.state.real_step == 0:
            actions = self.bid_policy()
        elif self.state.step < 0:
            actions = self.factory_placement_policy()
        else:
            actions = self.act()

        return process_action(actions)