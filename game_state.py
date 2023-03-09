import sys
import copy
import queue
import numpy as np
from itertools import chain

import clux

from lux.utils import code_to_direction, next_move, valid, distance

class Unit:
    def __init__(self, obs, time, factory_loc_dict, is_my):
        self.update(obs, time)
        self.is_my = is_my
        # mother factory location
        if self.is_my:
            self.occupation = "NO"
            self.is_baby = True
            self.init_pos = self.pos
            self.mother_ship = factory_loc_dict[self.init_pos[0], self.init_pos[1]]
            self.mother_ship.kids.append(self)

    def update(self, obs, time):
        self.time = time
        self.unit_id = obs["unit_id"]
        self.power = obs["power"]
        self.unit_type = obs["unit_type"]
        self.pos = obs["pos"]
        self.cargo = obs["cargo"]
        self.action_queue = obs["action_queue"]

class Factory:
    def __init__(self, obs, state, time, is_my = False):
        self.update(obs, time)
        self.is_my = is_my
        self.rubble_queue = []
        self.rubble_queue_head = 0
        self.last_queue_shuffle = 0
        self.state = state
        self.kids = []

    def update(self, obs, time):
        self.time = time
        self.unit_id = obs["unit_id"]
        self.strain_id = obs["strain_id"]
        self.power = obs["power"]
        self.pos = obs["pos"]
        self.cargo = obs["cargo"]

    def move_kids_to(self, step_mother):
        for kid in self.kids:
            kid.mother_ship = step_mother
            step_mother.kids.append(kid)
            kid.init_pos = step_mother.pos  # Brainwashing :)
            kid.occupation = "NO"

    def sort_rubble_queue(self):
        """sort function: dist * K + rubble_value"""
        K = 3.5
        self.rubble_queue.sort(key=lambda triple: triple[1] * K + triple[2])

    def next_rubble(self):
        # move head to the start of the queue so some unconsumed rubble can be consumed.
        if self.time - self.last_queue_shuffle > 150:
            self.last_queue_shuffle = self.time
            self.rubble_queue_head = 0

        while(True):
            if self.rubble_queue_head >= len(self.rubble_queue):
                return ((0,0),0,1)
            rubble = self.rubble_queue[self.rubble_queue_head]
            if self.state.rubble[rubble[0]] > 0:
                return rubble[0], rubble[1], self.state.rubble[rubble[0]]  # always returns current rubble state
            # skip rubble-less tile
            self.rubble_queue_head+=1

    def consume_rubble(self):
        self.rubble_queue_head += 1

def valid_loc(loc,x,y):
    return ( loc[0]+x >= 0 and loc[0]+x < 48 and loc[1]+y >= 0 and loc[1]+y < 48 )

def q_move(loc, dist, x, y):
    return ( np.array([loc[0]+x,loc[1]+y]), np.array([dist[0]+x, dist[1]+y]) )

# TODO this could be a c++ method
def build_distance_map(map, object_marker):
    EMPTY = -100

    dist_map = np.full((48,48,2), EMPTY, dtype=int)

    object_locations = np.argwhere(map >= object_marker)
    #print(object_locations, file=sys.stderr)

    q = queue.Queue()

    for loc in object_locations:
        q.put((loc, np.array([0,0])))

    while not q.empty():
        loc, dist = q.get()
        # If the distance is already calculated.
        if dist_map[loc[0],loc[1]][0] != EMPTY:
            continue

        dist_map[loc[0],loc[1]] = dist

        for x,y in [[0, -1], [1, 0], [0, 1], [-1, 0]]:
            if valid_loc(loc,x,y):
                q.put( q_move(loc,dist,x,y) )

    #np.set_printoptions(threshold=np.inf)
    #print(map, file=sys.stderr)
    #print(dist_map, file=sys.stderr)

    return dist_map

class GameState:

    def __init__(self, obs, cfg):
        # These variables are constant during the game or only appear in initial obs.
        self.me = obs.player
        self.him = "player_1"
        if self.me == self.him:
            self.him = "player_0"

        self.cfg = cfg

        self.ice = np.array(obs.obs["board"]["ice"])
        self.ore = np.array(obs.obs["board"]["ore"])

        self.ice_distance = build_distance_map(self.ice, 1)
        self.ore_distance = build_distance_map(self.ore, 1)
        # TODO for rubble,oponent lichen.. but this changes~~

        self.factories_per_team = obs.obs["board"]["factories_per_team"]

        # Initial set of variables. Later they come as a diff
        self.board = obs.obs["board"]
        self.rubble = np.array( self.board["rubble"])
        self.lichen = np.array( self.board["lichen"])
        self.lichen_strains = np.array( self.board["lichen_strains"])

        #self.rubble_distance = build_distance_map(self.rubble, 1)

        self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        self.units = dict()
        self.factories = dict()

        self.his_units = dict()
        self.his_factories = dict()

        self.no_go_map = np.zeros((48, 48))

        self.previous_state = None

        self.clux = clux.CLux(self.ice, self.ore)

        self.set_variable_obs(obs)

    def update(self, obs):
        self.previous_state = copy.copy(self)
        self.set_variable_obs(obs)

    def update_board(self, board, update_vec):
        for loc, value in update_vec.items():
            x_str, y_str = loc.split(",")
            x = int(x_str)
            y = int(y_str)
            board[x,y] = value

    def process_units(self, units_data, units, is_my = True):
        for unit_id in units_data.keys():
            if unit_id in units:
                units[unit_id].update(units_data[unit_id], self.real_step)
            else:
                units[unit_id] = Unit(units_data[unit_id], self.real_step, self.factory_loc_dict, is_my)
        delete_keys = []
        #self.unit_locs = dict()
        for unit_id in units.keys():
            unit = units[unit_id]
            if unit.time != self.real_step:
                delete_keys.append(unit_id)
            # else:
            #     self.unit_locs[(unit.pos[0],unit.pos[1])] = unit
        for key in delete_keys:
            del units[key]
            #print("del:", key, file=sys.stderr)

    #TODO slow
    def create_units_map(self, units, his_units):
        """
        0 - unit is there right now
        1 - most likely next move location based on action queue (only if moves)
        2 - potential next move location if queue changes (only if changes)
        """
        map = [[[] for i in range(48)] for j in range(48)]
        for unit in chain(units.values(), his_units.values()):
            map[unit.pos[0]][unit.pos[1]].append((0, unit))

            move_code = next_move(unit)

            for dir_code, dir in [(1,(0,-1)), (2,(1,0)), (3,(0,1)), (4,(-1,0))]:
                collision_code = 2
                if dir_code == move_code:
                    collision_code = 1
                loc = (unit.pos[0]+dir[0],unit.pos[1]+dir[1])
                if valid(*loc):
                    map[loc[0]][loc[1]].append((collision_code, unit))

        return map

    def build_no_go_map(self):
        """Opponent factories"""
        for factory in self.his_factories.values():
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    self.no_go_map[factory.pos[0] + i, factory.pos[1] + j] = 1

    def divide_and_conquer_rubble(self):
        # TODO use distance to his factories
        for i in range(self.rubble.shape[0]):
            for j in range(self.rubble.shape[1]):
                min_dist = 100
                closest_factory = None
                for factory in chain(self.factories.values(), self.his_factories.values()):
                    dist = distance( factory.pos, (i,j))
                    if dist < min_dist:
                        min_dist = dist
                        closest_factory = factory
                if self.rubble[i,j] <= 0 and min_dist > 3:  # Even if no rubble add close empty tiles. As someone could add dirt.
                    continue
                if closest_factory.is_my:
                    closest_factory.rubble_queue.append(((i,j), min_dist, self.rubble[i,j] ))

        for factory_id, factory in self.factories.items():
            factory.sort_rubble_queue()

    def set_variable_obs(self, obs):

        # TODO remove this once converted in controller.
        self.obs = obs.obs

        # from 0
        self.real_step = obs.step
        # from -7 to 1000
        self.step = obs.obs["real_env_steps"]

        units_data = obs.obs["units"][self.me]
        self.process_units( units_data, self.units )

        his_units_data = obs.obs["units"][self.him]
        self.process_units( his_units_data, self.his_units, is_my = False)

        self.units_map = self.create_units_map(self.units, self.his_units)

        factories_data = obs.obs["factories"][self.me]
        for factory_id in factories_data.keys():
            if factory_id in self.factories:
                self.factories[factory_id].update(factories_data[factory_id], self.real_step)
                #print("new_unit", factories_data[factory_id], file=sys.stderr)
            else:
                self.factories[factory_id] = Factory(factories_data[factory_id], self, self.real_step, is_my = True)

        delete_keys = []
        for fac_id in self.factories.keys():
            fac = self.factories[fac_id]
            if fac.time != self.real_step:
                delete_keys.append(fac_id)
        for key in delete_keys:
            death_mother = self.factories[key]
            del self.factories[key]
            death_mother.move_kids_to(next(iter(self.factories.values())))

        his_factories_data = obs.obs["factories"][self.him]
        for factory_id in his_factories_data:
            if factory_id in self.his_factories:
                self.his_factories[factory_id].update(his_factories_data[factory_id], self.real_step)
            else:
                self.his_factories[factory_id] = Factory(his_factories_data[factory_id], self, self.real_step)
        # Don't care about their death yet.

        if  len(obs.obs["teams"]) > 0:
            self.my_team = obs.obs["teams"][self.me]
            self.his_team = obs.obs["teams"][self.him]

        # print("step:", self.real_step, self.board.keys(), file=sys.stderr)

        if self.real_step > 0:
            self.board = obs.obs["board"]

            self.update_board( self.rubble, self.board["rubble"])
            self.update_board( self.lichen, self.board["lichen"])
            self.update_board( self.lichen_strains, self.board["lichen_strains"])

            #self.rubble_distance = build_distance_map(self.rubble, 1)

            if "valid_spawns_mask" in self.board:
                #print(self.board["valid_spawns_mask"], file=sys.stderr)
                self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        if self.step == 0:
            self.divide_and_conquer_rubble()
            self.factory_loc_dict = dict()
            self.build_no_go_map()
            for factory in self.factories.values():
                self.factory_loc_dict[(factory.pos[0], factory.pos[1])] = factory

        self.bonus_time_left = obs.remainingOverageTime

        self.clux.update_rubble(self.rubble)

