import sys
import copy
import queue
import numpy as np
from itertools import chain
from scipy.ndimage import gaussian_filter
from PIL import Image
from scipy.signal import convolve2d

import clux

from lux.utils import code_to_direction, next_move, valid, distance

def equal_queues(q1, q2):
    if len(q1) != len(q2):
        return False
    for x,y in zip(q1,q2):
        if not np.array_equal(x,y):
            return False
    return True

class Unit:
    def __init__(self, obs, time, factory_loc_dict, is_my):
        self.failed_actions = False
        self.update(obs, time)
        self.is_my = is_my
        # mother factory location
        if self.is_my:
            self.occupation = "NONE"
            self.is_baby = True
            self.init_pos = self.pos
            self.mother_ship = factory_loc_dict[self.init_pos[0], self.init_pos[1]]
            self.mother_ship.kids.append(self)

    def update(self, obs, time):
        self.time = time
        self.unit_id = obs["unit_id"]
        self.power = obs["power"]
        self.unit_type = obs["unit_type"]
        self.is_heavy = self.unit_type == "HEAVY"
        self.pos = obs["pos"]
        self.cargo = obs["cargo"]
        self.action_queue = obs["action_queue"]

        if self.failed_actions:
            self.action_queue = []


class Factory:
    def __init__(self, obs, state, time, is_my = False):
        self.update(obs, time)
        self.is_my = is_my
        if is_my:
            self.rubble_queue = []
            self.rubble_queue_head = 0
            self.last_queue_shuffle = 0
            self.state = state
            self.kids = []
            self.heavy_count = 0
            self.light_count = 0

    def update(self, obs, time):
        self.time = time
        self.unit_id = obs["unit_id"]
        self.strain_id = obs["strain_id"]
        self.power = obs["power"]
        self.pos = obs["pos"]
        self.pos_x = obs["pos"][0]
        self.pos_y = obs["pos"][1]
        self.cargo = obs["cargo"]

    def move_kids_to(self, step_mother):
        for kid in self.kids:
            kid.mother_ship = step_mother
            step_mother.kids.append(kid)
            kid.init_pos = step_mother.pos  # Brainwashing :)
            kid.action_queue = []
            kid.occupation = "NERVER"


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

def gaussian_kernel(size, sigma = 1, hole = False):

    kernel = np.zeros((size, size))
    center = size // 2
    kernel[center, center] = 1

    kernel = gaussian_filter(kernel, sigma=sigma)

    # hole 3x3 that factory covers
    if hole:
        for i in range(center-1,center+2):
            for j in range (center-1,center+2):
                kernel[i,j] = 0

    kernel /= kernel.sum()
    return kernel

def convolution(array, kernel_size, sigma = 1, kernel_hole = False , multi = 1):

    kernel = gaussian_kernel(kernel_size, sigma = sigma, hole=kernel_hole)
    p = kernel_size // 2
    padding = ((p, p), (p, p))

    convolved_array = convolve2d(np.pad(array, padding, mode='constant', constant_values=0), kernel, mode='valid')
    if multi > 1:
        for i in range(multi-1):
            convolved_array = convolve2d(
                np.pad(convolved_array, padding, mode='constant', constant_values=0), kernel, mode='valid')

    print("conv. array_size:", (convolved_array.max() - convolved_array.min()), " ", convolved_array.shape, file=sys.stderr)

    # Normalize pixel values between 0 and 1
    normalized_array = 100 * (convolved_array - convolved_array.min()) / (convolved_array.max() - convolved_array.min())

    return normalized_array

def numpy_to_img( array ):
    normalized_array = 255 * (array - array.min()) / (array.max() - array.min())
    uint8_array = np.uint8(normalized_array.T)
    image = Image.fromarray(uint8_array, mode='L')
    image.save('X.png')


def init_convolutions(state, rubble, ore, factories, his_factories):
    print( state.step, "rubble", file=sys.stderr)
    r = convolution(rubble, 15, kernel_hole = True)
    print("ore", file=sys.stderr)
    o = convolution(ore, 9, sigma = 5, multi = 5)

    mf = factory_map = np.zeros((48,48))
    for factory in factories.values():
        factory_map[factory.pos[0], factory.pos[1]] = 1

    hf = his_factory_map = np.zeros((48, 48))
    for factory in his_factories.values():
        his_factory_map[factory.pos[0], factory.pos[1]] = 1

    print("mf", len(factories), "map ", factory_map[7][7], file=sys.stderr)
    if len(factories) > 0:
        mf = convolution(factory_map, 7, sigma=5, multi=10)
    print("hf", len(his_factories), "map ", his_factory_map[7][7], file=sys.stderr)
    if len(his_factories) > 0:
        hf = convolution(his_factory_map, 7, sigma=5, multi=10)

    # minimizing score. Black is better

    best_places = 30 * r - 30 * o + 1 * mf - 1 * hf
    # numpy_to_img(best_places)

    return best_places


class GameState:

    def __init__(self, cfg):
        self.cfg = cfg

    def init_update(self, obs, agent = "", real_step_ = 0):
        if agent == "":
            # code from main
            self.me = obs.player
            real_step = obs.real_step
            # TODO skipped for now
            # remaining_time = obs.remainingOverageTime
            obs = obs.obs
        else:
            self.me = agent
            real_step = real_step_

        self.him = "player_1"
        if self.me == self.him:
            self.him = "player_0"


        self.ice = np.array(obs["board"]["ice"])
        self.ore = np.array(obs["board"]["ore"])

        self.ice_distance = build_distance_map(self.ice, 1)

        self.factories_per_team = obs["board"]["factories_per_team"]

        # Initial set of variables. Later they come as a diff
        self.board = obs["board"]
        self.rubble = np.array(self.board["rubble"])
        self.lichen = np.array(self.board["lichen"])
        self.lichen_strains = np.array(self.board["lichen_strains"])

        # self.rubble_distance = build_distance_map(self.rubble, 1)

        self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        self.units = dict()
        self.factories = dict()

        self.his_units = dict()
        self.his_factories = dict()

        self.no_go_map = np.zeros((48, 48))
        self.chick_chick_locs = []

        self.previous_state = None

        self.has_lichen = False
        self.he_has_lichen = False

        self.clux = clux.CLux(self.ice, self.ore, self.factories_per_team)

        self.set_variable_obs(obs, real_step)

    def update(self, obs, agent = "", real_step = 0):
        # self.previous_state = copy.copy(self)
        if agent == "":
            self.set_variable_obs(obs.obs, obs.real_step)
        else:
            self.set_variable_obs(obs, real_step)

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
            self.clux.remove_zombie_unit(key)
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
        self.no_go_map = np.zeros((48, 48))
        for factory in self.his_factories.values():
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    self.no_go_map[factory.pos[0] + i, factory.pos[1] + j] = 1

    def build_chick_chick_vec(self):
        """Opponent factories"""
        for factory in self.factories.values():
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    self.chick_chick_locs.append((factory.pos[0] + i, factory.pos[1] + j))


    def update_has_lichen(self):
        is_my_strain = {}
        for factory in self.factories.values():
            is_my_strain[factory.strain_id] = True
        for factory in self.his_factories.values():
            is_my_strain[factory.strain_id] = False

        unique_items, item_counts = np.unique(self.lichen_strains, return_counts=True)

        for strain_id, is_my in is_my_strain.items():
            if strain_id in unique_items:
                if is_my:
                    self.has_lichen = True
                else:
                    self.he_has_lichen = True


    def set_variable_obs(self, obs, real_step):

        # from 0
        self.real_step = real_step

        # from -7 to 1000
        self.step = obs["real_env_steps"]

        units_data = obs["units"][self.me]
        self.process_units( units_data, self.units )

        his_units_data = obs["units"][self.him]
        self.process_units( his_units_data, self.his_units, is_my = False)

        # self.units_map = self.create_units_map(self.units, self.his_units)

        factories_data = obs["factories"][self.me]
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
            self.clux.remove_zombie_factory(key)
            death_mother.move_kids_to(next(iter(self.factories.values())))

        his_factories_data = obs["factories"][self.him]
        for factory_id in his_factories_data:
            if factory_id in self.his_factories:
                self.his_factories[factory_id].update(his_factories_data[factory_id], self.real_step)
            else:
                self.his_factories[factory_id] = Factory(his_factories_data[factory_id], self, self.real_step)
        delete_keys = []
        for fac_id in self.his_factories.keys():
            fac = self.his_factories[fac_id]
            if fac.time != self.real_step:
                delete_keys.append(fac_id)
        for key in delete_keys:
            del self.his_factories[key]
            self.clux.remove_zombie_factory(key)
        if len(delete_keys) > 0:
            self.build_no_go_map()

        # print("______", real_step)
        # print(obs)

        if  len(obs["teams"]) > 0:
            self.my_team = obs["teams"][self.me]
            self.his_team = obs["teams"][self.him]

        # print("step:", self.real_step, self.board.keys(), file=sys.stderr)

        if self.real_step > 0:
            self.board = obs["board"]

            if isinstance(self.board["rubble"], np.ndarray):
                self.rubble = self.board["rubble"]
                self.lichen = self.board["lichen"]
                self.lichen_strains = self.board["lichen_strains"]
            else:
                self.update_board( self.rubble, self.board["rubble"])
                self.update_board( self.lichen, self.board["lichen"])
                self.update_board( self.lichen_strains, self.board["lichen_strains"])

            #self.rubble_distance = build_distance_map(self.rubble, 1)

            if "valid_spawns_mask" in self.board:
                #print(self.board["valid_spawns_mask"], file=sys.stderr)
                self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        # if self.step < 0:
        #     init_convolution = init_convolutions(self, self.rubble, self.ore, self.factories, self.his_factories)
        #     self.clux.update_factory_init_convolution(init_convolution)

        if self.step == 0:
            self.factory_loc_dict = dict()
            self.build_no_go_map()
            self.build_chick_chick_vec()
            for factory in self.factories.values():
                self.factory_loc_dict[(factory.pos[0], factory.pos[1])] = factory

        self.clux.update_rubble(self.rubble)
        self.clux.update_lichen(self.lichen, self.lichen_strains)

        self.update_has_lichen()

        for unit in chain(self.units.values(), self.his_units.values()):
            mother_ship_id = ""
            if unit.is_my:
                mother_ship_id = unit.mother_ship.unit_id
            self.clux.update_unit( unit.unit_id, unit.unit_type == "HEAVY", unit.is_my,
                                       unit.power, unit.pos[0], unit.pos[1], unit.cargo,
                                       unit.action_queue, mother_ship_id)

        for f in chain(self.factories.values(), self.his_factories.values()):
            self.clux.update_factory( f.unit_id, f.strain_id, f.is_my,
                                      f.power, f.pos[0], f.pos[1], f.cargo)

        # Called last.
        self.clux.update_assorted( self.real_step, self.step, self.me == "player_0")