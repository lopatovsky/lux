import sys

import numpy as np
import torch as th

from lux.kit import process_action
from lux.utils import code_to_direction, next_move, valid

from wrappers import LuxObservationWrapper, LuxController

from game_state import GameState

def distance(x,y):
    return abs(x[0] - y[0]) + abs(x[1] - y[1])


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
                    if not valid(i, j):
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

    def ad_hoc_dist(self):
        rand_num = np.random.randint(low=1, high=5)
        x,y = code_to_direction(rand_num)
        return (x*2, y*2)

    def closest_home(self, home, pos):
        new_home = home
        dist = 100

        for x in [-1,0,1]:
            for y in [-1,0,1]:
                d = distance(pos, (home[0]+x,home[1]+y))
                if d < dist:
                    dist = d
                    new_home = (home[0]+x,home[1]+y)

        return new_home

    def mine_resource_action(self, unit, resource_dist_map, resource_code):
        actions = []

        if resource_dist_map is None:
            dist = self.ad_hoc_dist()
        else:
            dist = resource_dist_map[unit.pos[0],unit.pos[1]]

        target_pos = ((unit.pos[0] - dist[0]), (unit.pos[1] - dist[1]))

        home_pos = self.closest_home(unit.init_pos, target_pos)
        # home_pos = unit.init_pos


        #print(unit.unit_id, "home-center: ", unit.init_pos, "home new:", home_pos, "pos", unit.pos, "target-pos:", target_pos, file=sys.stderr )

        actions.extend(self.move_dist(dist[0], dist[1]))
        actions.extend(self.dig_action(10))
        actions.extend(self.move_dist(target_pos[0] - home_pos[0], target_pos[1] - home_pos[1]))
        actions.extend(self.transfer_action(resource_code))

        #print(unit.pos, actions)

        # TODO check current power
        power = 70
        if unit.unit_type == "HEAVY":
            power = 800

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
        return self.mine_resource_action(unit, None, 0)
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

    def stupid_action(self, unit):
        if len(unit.action_queue) == 0:
            return False

        next_action = unit.action_queue[0]
        px = unit.pos[0]
        py = unit.pos[1]
        state = self.state

        if next_action[0] == 0:  # move
            move_dir = code_to_direction(next_action[1])
            move_loc = (px + move_dir[0], py + move_dir[1])
            if not valid(*move_loc):
                return True

            # if unit.unit_id == "unit_39":
            #     print(state.step, ":", px,py, move_dir, move_loc, file=sys.stderr )

        elif next_action[0] == 3:
            # TODO don't dig own lichen
            if state.ice[px,py] + state.ore[px,py] + state.rubble[px,py] + state.lichen[px,py] == 0:
                return True

        return False

    def win_collision(self, unit, c_unit):
        # TODO check for energy consumption of the current move & if power are the same
        if unit.unit_type == c_unit.unit_type:
            return unit.power > c_unit.power
        return unit.unit_type < c_unit.unit_type  # "H" < "L"

    def is_safe(self, unit, colliding_units):
        for code, c_unit in colliding_units:
            if unit == c_unit:
                continue
            if not self.win_collision(unit, c_unit):
                return False
        return True

    def resolve_collisions(self, unit):
        move_code = next_move(unit)
        move_dir = code_to_direction(move_code)
        move_pos = ( unit.pos[0] + move_dir[0], unit.pos[1] + move_dir[1])

        potential_collisions = self.state.units_map[move_pos[0]][move_pos[1]]
        safe = True
        for code, c_unit in potential_collisions:
           if unit == c_unit:
               continue
           if not self.win_collision(unit,c_unit):
               # Weaker must run away.
               safe = False

        # if unit.unit_type == "HEAVY":
        #    print( unit.unit_id, safe, file=sys.stderr )

        if not safe:
            safe_dir_codes = []
            for code, dir in [(1, (0, -1)), (2, (1, 0)), (3, (0, 1)), (4, (-1, 0))]:
                loc = unit.pos[0] + dir[0], unit.pos[1] + dir[1]
                if valid(*loc) and self.is_safe(unit, self.state.units_map[loc[0]][loc[1]]):
                    safe_dir_codes.append(code)
            length = len(safe_dir_codes)
            if length == 0:
                print("trapped", file=sys.stderr)
                # TODO still possible to optimize to go places with higher collision code.
                return [np.array([0, np.random.randint(low=0, high=5), 0, 0, 0, 1])]
            else:
                # TODO movement is valid?
                return [np.array([0, safe_dir_codes[np.random.randint(low=0, high=length)], 0, 0, 0, 1])]
        return []

    # TODO move this to controller?
    def rule_based_actions(self):
        lux_action = dict()

        units = self.state.units
        factories = self.state.factories

        for unit_id in units.keys():
            if self.stupid_action(units[unit_id]):
                units[unit_id].action_queue = []


        for unit_id in units.keys():
            unit = units[unit_id]

            if len(unit.action_queue) == 0:
                if unit.unit_type == 'HEAVY':
                    lux_action[unit_id] = self.mine_ice_action(unit)
                else:
                    # TODO query home factory
                    #if np.random.rand() < 1.2:
                    lux_action[unit_id] = self.remove_rubble_action(unit)
                    #else: lux_action[unit_id] = self.mine_ore_action(unit)

        # Collisions
        for unit_id in units.keys():
            action = self.resolve_collisions(units[unit_id])
            if len(action) != 0:
                lux_action[unit_id] = action

        # FACTORIES ACTION

        # Create a heavy unit.
        if len(units) == 0:
            for factory_id in factories.keys():
                lux_action[factory_id] = 1

        # Create a light unit
        else:
            for factory_id in factories.keys():
                factory = factories[factory_id]
                if factory.cargo["metal"] < 10:
                    continue
                # todo don't create if there is something standing
                lux_action[factory_id] = 0

        for factory_id in self.state.factories.keys():
            factory = self.state.factories[factory_id]
            if factory.cargo["water"] > 6*(1000 - self.state.step):
                lux_action[factory_id] = 2  # water and grow lichen at the end of the game

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