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

        self.units = dict()
        self.factories = dict()

    def bid_policy(self):
        return dict(faction="AlphaStrike", bid=1)

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

            closest_ice = self.state.ice_distance[x,y]
            ice_dist = abs(closest_ice[0]) + abs(closest_ice[1])

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

            if ice_dist == 2 and min_rubble < 0:
                min_rubble -= 5  # bonus for short path to ice TODO temporary need

            if has_ice and min_rubble < lowest_rubble:
                lowest_rubble = min_rubble
                pos = spawn_loc

        metal = min(state.cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["metal"])
        water = min(state.cfg.INIT_WATER_METAL_PER_FACTORY, my_obs["water"])
        return dict(spawn=pos, metal=metal, water=water)

    def act(self):

        # obs = LuxObservationWrapper.convert_obs(self.state)
        #
        # obs = th.from_numpy(obs).float()
        # with th.no_grad():
        #
        #     # to improve performance, we have a rule based action mask generator for the controller used
        #     # which will force the agent to generate actions that are valid only.
        #     action_mask = (
        #         th.from_numpy(self.controller.action_masks())
        #         .unsqueeze(0)
        #         .bool()
        #     )
        #
        #     # SB3 doesn't support invalid action masking. So we do it ourselves here
        #     features = self.ppo_model.policy.features_extractor(obs.unsqueeze(0))
        #     x = self.ppo_model.policy.mlp_extractor.shared_net(features)
        #     logits = self.ppo_model.policy.action_net(x)  # shape (1, N) where N=12 for the default controller
        #
        #     logits[~action_mask] = -1e8  # mask out invalid actions
        #     dist = th.distributions.Categorical(logits=logits)
        #     actions = dist.sample().cpu().numpy()  # shape (1, 1)
        #
        # # use our controller which we trained with in train.py to generate a Lux S2 compatible action
        # lux_action = self.controller.action_to_lux_action(actions[0])
        #
        #
        # # grows lichen
        # for unit_id in self.state.factories.keys():
        #     factory = self.state.factories[unit_id]
        #     if 1000 - self.state.step < 201 and factory["cargo"]["water"] > 200:
        #         lux_action[unit_id] = 2  # water and grow lichen at the very end of the game
        #
        # return lux_action

        return self.rule_based_actions()

    def move_dist(self, x, y):
        """Move directions: [0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]"""
        actions = []
        if x > 0:
            actions.append(np.array([0, 4, 0, 0, 0, x]))
        elif x < 0:
            actions.append(np.array([0, 2, 0, 0, 0, -x]))
        if y > 0:
            actions.append(np.array([0, 1, 0, 0, 0, y]))
        elif y < 0:
            actions.append(np.array([0, 3, 0, 0, 0, -y]))
        return actions

    def dig_action(self, n):
        return [np.array([3, 0, 0, 0, 0, n])]

    def recharge_action(self, till_capacity):
        return [np.array([5, 0, 0, till_capacity, 0, 1])]

    def pick_up_action(self, resource_code, pick_up_amount ):
        return [np.array([2, 0, resource_code, pick_up_amount, 0, 1])]

    def transfer_action(self, resource_code):
        """material = (0 = ice, 1 = ore, 2 = water, 3 = metal, 4 power)"""
        transfer_dir = 0
        return [np.array([1, transfer_dir, resource_code, self.state.cfg.max_transfer_amount, 0, 1])]

    def mine_resource_action(self, unit, resource_dist_map, resource_code):
        actions = []

        dist = resource_dist_map[unit.pos[0],unit.pos[1]]
        home_pos = unit.init_pos

        actions.extend(self.move_dist(dist[0], dist[1]))
        actions.extend(self.dig_action(5))
        actions.extend(self.move_dist(home_pos[0] - (unit.pos[0] + dist[0]), home_pos[1] - (unit.pos[1] + dist[1])))
        actions.extend(self.transfer_action(resource_code))

        # TODO check current power
        power = 50
        if unit.unit_type == "HEAVY":
            power = 400

        actions.extend(self.pick_up_action(4, power)) # 4: power

        return actions
    def mine_ice_action(self, unit):
        return self.mine_resource_action(unit, self.state.ice_distance, 0)  # 0: ice

        # find closest ice
        # if home: charge till possible to do action -how to
        # get_move_actions to get there and energy?
        # dig -- how many times? + e
        # get home + e
        # if not enough energy:
        # go home to charge first. # need action or just stay

    def mine_ore_action(self, unit):
        return self.mine_resource_action(unit, self.state.ore_distance, 1)
        #rand_num = np.random.randint(low=1, high=5)

    def remove_rubble_action(self, unit):
        return self.mine_resource_action(unit, self.state.rubble_distance, 1)
    # actions for robots:
    # - dig ice
    # - dig metal
    # - dig rubbles
    # - dig lichen of opponent
    # special -> avoid crashes or kill opponent

    # actions for factories:
    # - create light robot
    # - create heavy robot
    # - water lichen

    def remove_rubble(self, unit):
        pass
        # basically almost mining

    # TODO move this to controller?
    def rule_based_actions(self):
        lux_action = dict()

        units = self.state.units
        factories = self.state.factories

        for unit_id in units.keys():
            unit = units[unit_id]

            # if collision_risk(unit):


            if len(unit.action_queue) == 0:
                if unit.unit_type == 'HEAVY':
                    lux_action[unit_id] = self.mine_ice_action(unit)
                else:
                    lux_action[unit_id] = self.remove_rubble_action(unit)

        # FACTORIES ACTION

        # Create a heavy unit.
        if len(units) == 0:
            for factory_id in factories.keys():
                lux_action[factory_id] = 1

        # Create a light unit
        else:
            for factory_id in factories.keys():
                factory = factories[factory_id]
                if factory.cargo["metal"] < 45:
                    continue
                # todo don't create if there is something standing
                lux_action[factory_id] = 0

        for factory_id in self.state.factories.keys():
            factory = self.state.factories[factory_id]
            if factory.cargo["water"] > 6*(1000 - self.state.step):
                lux_action[factory_id] = 2  # water and grow lichen at the very end of the game

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