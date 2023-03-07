import sys
import copy
import queue
import numpy as np

from lux.config import EnvConfig

class Unit:
    def __init__(self, obs):
        self.update(obs)

        # mother factory location
        self.init_pos = self.pos

    def update(self, obs):
        self.unit_id = obs["unit_id"]
        self.power = obs["power"]
        self.unit_type = obs["unit_type"]
        self.pos = obs["pos"]
        self.cargo = obs["cargo"]
        self.action_queue = obs["action_queue"]

class Factory:
    def __init__(self, obs):
        self.update(obs)

    def update(self, obs):
        self.unit_id = obs["unit_id"]
        self.strain_id = obs["strain_id"]
        self.power = obs["power"]
        self.pos = obs["pos"]
        self.cargo = obs["cargo"]

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

    def __init__(self, obs):
        # These variables are constant during the game or only appear in initial obs.
        self.me = obs["player"]
        self.him = "player_1"
        if self.me == self.him:
            self.him = "player_0"

        self.cfg = EnvConfig.from_dict(obs['info']['env_cfg'])

        self.ice = np.array(obs["obs"]["board"]["ice"])
        self.ore = np.array(obs["obs"]["board"]["ore"])

        self.ice_distance = build_distance_map(self.ice, 1)
        self.ore_distance = build_distance_map(self.ore, 1)
        # TODO for rubble,oponent lichen.. but this changes~~

        self.factories_per_team = obs["obs"]["board"]["factories_per_team"]

        # Initial set of variables. Later they come as a diff
        self.board = obs["obs"]["board"]
        self.rubble = np.array( self.board["rubble"])
        self.lichen = np.array( self.board["lichen"])
        self.lichen_strains = np.array( self.board["lichen_strains"])

        self.rubble_distance = build_distance_map(self.rubble, 1)

        self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        self.previous_state = None
        self.set_variable_obs(obs)

        self.units = dict()
        self.factories = dict()

    def update(self, obs):
        self.previous_state = copy.copy(self)
        self.set_variable_obs(obs)

    def update_board(self, board, update_vec):
        for loc, value in update_vec.items():
            x_str, y_str = loc.split(",")
            x = int(x_str)
            y = int(y_str)
            board[x,y] = value


    def set_variable_obs(self, obs):

        # TODO remove this once converted in controller.
        self.obs = obs["obs"]

        # from 0
        self.real_step = obs["step"]
        # from -7 to 1000
        self.step = obs["obs"]["real_env_steps"]

        units_data = obs["obs"]["units"][self.me]

        for unit_id in units_data.keys():
            if unit_id in self.units:
                self.units[unit_id].update(units_data[unit_id])
            else:
                self.units[unit_id] = Unit(units_data[unit_id])

        factories_data = obs["obs"]["factories"][self.me]
        for factory_id in factories_data.keys():
            if factory_id in self.factories:
                self.factories[factory_id].update(factories_data[factory_id])
                #print("new_unit", factories_data[factory_id], file=sys.stderr)
            else:
                self.factories[factory_id] = Factory(factories_data[factory_id])

        self.his_units = obs["obs"]["units"][self.him]
        self.his_factories = obs["obs"]["factories"][self.him]

        if  len(obs["obs"]["teams"]) > 0:
            self.my_team = obs["obs"]["teams"][self.me]
            self.his_team = obs["obs"]["teams"][self.him]

        # print("step:", self.real_step, self.board.keys(), file=sys.stderr)

        # todo later come updates not whole field
        if self.real_step > 0:
            self.board = obs["obs"]["board"]

            #print("rubble: ", self.board["rubble"], file=sys.stderr)

            self.update_board( self.rubble, self.board["rubble"])
            self.update_board( self.lichen, self.board["lichen"])
            self.update_board( self.lichen_strains, self.board["lichen_strains"])

            #self.rubble_distance = build_distance_map(self.rubble, 1)

            if "valid_spawns_mask" in self.board:
                #print(self.board["valid_spawns_mask"], file=sys.stderr)
                self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        # obs global_id: 0 ?

        self.bonus_time_left = obs["remainingOverageTime"]
        self.reward = obs["reward"]  #?

