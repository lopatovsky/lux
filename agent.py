import sys
import math
import random

import numpy as np
import torch as th

from lux.kit import process_action
from lux.utils import code_to_direction, next_move, valid, distance

from wrappers import LuxObservationWrapper, LuxController

from game_state import GameState




def AdjToFactory(x, y, i, j):
    dist_x = abs(x - i)
    dist_y = abs(y - j)
    if dist_x == 2 and dist_y == 2: return False
    if dist_x == 2 or dist_y == 2: return True
    return False

class Agent:
    def __init__(self, ppo_model, state: GameState) -> None:
        np.random.seed(0)
        random.seed(0)
        self.ppo_model = ppo_model
        self.state = state

        # TODO why ObsWrapper is not init here
        self.controller = LuxController(self.state)

        self.units = dict()
        self.factories = dict()

    def bid_policy(self):
        return dict(faction="AlphaStrike", bid=2)

    def factory_placement_policy(self):
        """
        This policy will place a single factory with all the starting resources
        near a random ice tile and if possible close to an empty lot.
        """
        state = self.state
        pos = state.clux.place_factory()
        my_obs = state.my_team

        if my_obs["metal"] == 0:
            return dict()

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

    def compute_move_price(self, pos, move_actions):
        # TODO maybe the best possible path not some random.
        rubble = self.state.rubble
        no_go = self.state.no_go_map
        px = pos[0]
        py = pos[1]

        price = 0
        for action in move_actions:
            dir = code_to_direction(action[1])
            for i in range(action[5]):
                px += dir[0]
                py += dir[1]
                if not valid(px,py) or no_go[px,py]:
                    price += 1000
                price += 1 + (rubble[px,py] // 20)  # for small robot

        return price

    def move_dist(self, x, y):
        """Move directions: [0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]"""
        actions = []
        sub_moves = []
        if x > 0:
            sub_moves.append((4,x))
        elif x < 0:
            sub_moves.append((2,-x))
        if y > 0:
            sub_moves.append((1,y))
        elif y < 0:
            sub_moves.append((3,-y))

        split = []
        for code, x in sub_moves:
            div = x//3
            split.extend([[code, div], [code, div], [code, div + x%3]])

        random.shuffle(split)

        for code, x in split:
            if x > 0:
                actions.append(np.array([0, code, 0, 0, 0, x]))

        return actions

    def move_from_to(self, x, y):

        #return self.state.clux.shortest_path(x[0], x[1], y[0], y[1], False)

        return self.move_dist(x[0] - y[0], x[1] - y[1])

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

    def is_home(self, home, pos):
        return max(abs(home[0] - pos[0]) , abs(home[1] - pos[1])) <= 1

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

        dist = resource_dist_map[unit.pos[0],unit.pos[1]]

        target_pos = ((unit.pos[0] - dist[0]), (unit.pos[1] - dist[1]))

        home_pos = self.closest_home(unit.init_pos, target_pos)
        # home_pos = unit.init_pos


        #print(unit.unit_id, "home-center: ", unit.init_pos, "home new:", home_pos, "pos", unit.pos, "target-pos:", target_pos, file=sys.stderr )

        actions.extend(self.move_dist(dist[0], dist[1]))
        actions.extend(self.dig_action(10))
        actions.extend(self.move_from_to(target_pos, home_pos))
        actions.extend(self.transfer_action(resource_code))

        #print(unit.pos, actions)

        # TODO check current power
        power = 70
        if unit.unit_type == "HEAVY":
            power = 800

        actions.extend(self.pick_up_action(4, power)) # 4: power

        return actions


    def mine_ore_action(self, unit):
        return self.mine_resource_action(unit, self.state.ore_distance, 1)
        #rand_num = np.random.randint(low=1, high=5)

    def mine_ice_action(self, unit):
        return self.mine_resource_action(unit, self.state.ice_distance, 0)
        #rand_num = np.random.randint(low=1, high=5)

    def remove_rubble_action(self, unit):
        actions = []

        target_pos, total_dist , rubble_value = unit.mother_ship.next_rubble()

        was_baby = False
        if unit.is_baby:
            unit.is_baby = False
            was_baby = True
            actions.extend(self.pick_up_action(4, 99))  # New unit gets a lot of power because it's needed in beginning the most.

        dig_times = math.ceil(rubble_value / 2)  # LIGHT dig by 2
        energy_need = 5*dig_times + 2*total_dist

        if energy_need > unit.power and unit.power != 150 and not was_baby: # LIGHT unit battery capacity is 150
            home_pos = self.closest_home(unit.init_pos, unit.pos)
            actions.extend(self.move_from_to(unit.pos, home_pos))
            power_to_add = energy_need - unit.power + total_dist*2
            actions.extend(self.pick_up_action(4, power_to_add))  # 4: power
            return actions

        # accepts work on rubble removing.
        unit.mother_ship.consume_rubble()

        home_pos = self.closest_home(unit.init_pos, target_pos)

        if self.is_home(unit.init_pos, unit.pos) and not was_baby:
            actions.extend(self.pick_up_action(4, total_dist*2))
        actions.extend(self.move_from_to(unit.pos, target_pos))
        actions.extend(self.dig_action(dig_times))
        actions.extend(self.move_from_to(target_pos, home_pos))

        return actions

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
            if not valid(*move_loc) or state.no_go_map[move_loc]:
                return True

            # if unit.unit_id == "unit_39":
            #     print(state.step, ":", px,py, move_dir, move_loc, file=sys.stderr )

        elif next_action[0] == 3:
            # TODO don't dig own lichen
            if state.ice[px,py] + state.ore[px,py] + state.rubble[px,py] + state.lichen[px,py] == 0:
                return True

        return False

    def is_night(self):
        mod_step = self.state.step % 50
        return mod_step >= 30

    def insufficient_power(self, unit):
        # TODO this is only for light units. Sometimes may be unnecessery restrictive.
        return unit.power < 6

    def win_collision(self, unit, move_code, c_unit, collision_code):
        # TODO check for energy consumption of the current move & if power are the same
        if unit.unit_type == c_unit.unit_type:
            if move_code == 0 and collision_code > 0:
                if c_unit.is_my or self.insufficient_power( unit ):
                    return True  # standing or weak unit is sacred
                return False
            elif collision_code == 0 and move_code > 0:
                if c_unit.is_my or self.insufficient_power( c_unit ): # never move into standing unit or weak unit
                    return False
                return True  # returning True means possible killing.
            else:
                return unit.power > c_unit.power
        return unit.unit_type < c_unit.unit_type  # "H" < "L"

    def is_safe(self, unit, move_code, colliding_units):

        #  if move_code zero than win collision not on power
        for collision_code, c_unit in colliding_units:
            if unit == c_unit:
                continue
            if not self.win_collision(unit, move_code, c_unit, collision_code):
                return False
        return True

    def is_small_risk(self, unit, move_code, colliding_units):
        min_code = 2
        for collision_code, c_unit in colliding_units:
            if unit == c_unit:
                continue
            if not self.win_collision(unit, move_code, c_unit, collision_code):
                min_code = min(min_code, collision_code)
        return min_code == 2

    def resolve_collisions(self, unit, lux_actions):
        # Either check unit's action queue, or the lux actions that are about to overwrite it.
        if unit.unit_id in lux_actions:
            u_actions = lux_actions[unit.unit_id]
            move_code = 0
            if len(u_actions) > 0 and u_actions[0][0] == 0:  # first action's first value is zero -> means moving action.
                move_code = u_actions[0][1]  # move code of first action
        else:
            move_code = next_move(unit)

        move_dir = code_to_direction(move_code)
        move_pos = (unit.pos[0] + move_dir[0], unit.pos[1] + move_dir[1])

        if self.is_safe(unit, move_code, self.state.units_map[move_pos[0]][move_pos[1]]):
            return []

        safe_dir_codes = []
        small_risk_dir_codes = []
        for code, dir in [(0, (0, 0)) ,(1, (0, -1)), (2, (1, 0)), (3, (0, 1)), (4, (-1, 0))]:
            loc = unit.pos[0] + dir[0], unit.pos[1] + dir[1]
            if valid(*loc) and not self.state.no_go_map[loc]:
                if self.is_safe(unit, code, self.state.units_map[loc[0]][loc[1]]):
                    safe_dir_codes.append(code)
                elif self.is_small_risk(unit, code, self.state.units_map[loc[0]][loc[1]]):
                    small_risk_dir_codes.append(code)
        length = len(safe_dir_codes)
        if length == 0:
            print("potentially trapped", file=sys.stderr)
            length = len(small_risk_dir_codes)
            if length == 0:
                print("trapped", file=sys.stderr)
                return [np.array([0, np.random.randint(low=0, high=5), 0, 0, 0, 1])]
            else:
                return [np.array([0, small_risk_dir_codes[np.random.randint(low=0, high=length)], 0, 0, 0, 1])]
        else:
            return [np.array([0, safe_dir_codes[np.random.randint(low=0, high=length)], 0, 0, 0, 1])]


    # TODO move this to controller?
    def rule_based_actions(self):
        lux_action = dict()

        clux = self.state.clux

        units = self.state.units
        factories = self.state.factories

        for unit_id in units.keys():
            if self.stupid_action(units[unit_id]):
                # sets action queue to empty so it is later pick-ed up as idle and get more meaningful activity.
                units[unit_id].action_queue = []

        has_lichen = self.state.has_lichen
        he_has_lichen = self.state.he_has_lichen


        for unit_id in units.keys():
            unit = units[unit_id]

            if len(unit.action_queue) == 0:

                # TODO orchestration by ML oracle.
                np_rand = np.random.rand()

                if unit.occupation == "NONE":
                    if unit.unit_type == 'HEAVY' and self.state.step < 10:
                        unit.occupation = 'ICE_MINER'
                    elif unit.unit_type == 'HEAVY':
                        unit.occupation = 'NERVER'
                    elif not he_has_lichen:  # LIGHT
                        if np_rand < 0.6:
                            unit.occupation = 'RUBBLE_EATER'
                        elif np_rand < 0.8:
                            unit.occupation = 'ORE_MINER'
                        else:
                            unit.occupation = 'NERVER'
                    else:
                        if np_rand < 0.5:
                            unit.occupation = "INNER_LICHEN_EATER"
                        else:
                            unit.occupation = "OUTER_LICHEN_EATER"

                if he_has_lichen and has_lichen and unit.occupation == 'ORE_MINER':
                    unit.occupation = 'INNER_LICHEN_EATER'
                if he_has_lichen and has_lichen and unit.occupation == 'RUBBLE_EATER' and np_rand < 0.01:
                    unit.occupation = 'OUTER_LICHEN_EATER'

                if self.state.step > 990 and (unit.occupation == 'NERVER' or unit.occupation == 'INNER_LICHEN_EATER'):
                    unit.occupation = 'HARAKIRI_SAMURAI'

                # TODO implement these:
                # so far inner lichen eater seem has a little value , but if big is there it would have.
                if unit.occupation == 'HARAKIRI_SAMURAI' or unit.occupation == 'NERVER':
                    unit.occupation = 'OUTER_LICHEN_EATER'


                if unit.occupation == "ICE_MINER":
                    lux_action[unit_id] = self.mine_ice_action(unit)  # clux.mine_ice_action(unit_id)
                elif unit.occupation == "ORE_MINER":
                    lux_action[unit_id] = clux.mine_ore_action(unit_id)
                elif unit.occupation == "RUBBLE_EATER":
                    lux_action[unit_id] = clux.remove_rubble_action(unit_id)
                elif unit.occupation == "INNER_LICHEN_EATER":
                    lux_action[unit_id] = clux.remove_lichen_action(unit_id, True)
                elif unit.occupation == "OUTER_LICHEN_EATER":
                    lux_action[unit_id] = clux.remove_lichen_action(unit_id, False)
                elif unit.occupation == "NERVER":  # pro-fighter
                    lux_action[unit_id] = clux.distract_oponent_action(unit_id, False)
                elif unit.occupation == "HARAKIRI_SAMURAI":
                    lux_action[unit_id] = clux.suicide_action(unit_id, False)

        occ_counts = {}
        for unit in units.values():
            if unit.occupation in occ_counts:
                occ_counts[unit.occupation] +=1
            else:
                occ_counts[unit.occupation] = 0

        print(occ_counts, file=sys.stderr)

        # Collisions
        for unit_id in units.keys():
            action = self.resolve_collisions(units[unit_id], lux_action)
            if len(action) != 0:
                lux_action[unit_id] = action

        # FACTORIES ACTION

        # creates a heavy unit.
        if len(units) == 0:
            for factory_id in factories.keys():
                lux_action[factory_id] = 1

        # creates a light unit
        else:
            for factory_id in factories.keys():
                factory = factories[factory_id]
                if factory.cargo["metal"] < 10:
                    continue
                lux_action[factory_id] = 0

        # doesn't create unit if it is dangerous
        for factory_id, factory in factories.items():
            if factory_id in lux_action:
                if len(self.state.units_map[factory.pos[0]][factory.pos[1]]) > 0:
                    del lux_action[factory_id]

        for factory_id in self.state.factories.keys():
            factory = self.state.factories[factory_id]
            K = 6
            if factory.cargo["water"] > K *(1000 - self.state.step) + 20:
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