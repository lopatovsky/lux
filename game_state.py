import sys
import copy
import numpy as np

from lux.config import EnvConfig

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
        self.factories_per_team = obs["obs"]["board"]["factories_per_team"]

        # Initial set of variables. Later they come as a diff
        self.board = obs["obs"]["board"]
        self.rubble = np.array( self.board["rubble"])
        self.lichen = np.array( self.board["lichen"])
        self.lichen_strains = np.array( self.board["lichen_strains"])
        self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        self.previous_state = None
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


    def set_variable_obs(self, obs):

        # TODO remove this once converted in controller.
        self.obs = obs["obs"]

        # from 0
        self.real_step = obs["step"]
        # from -7 to 1000
        self.step = obs["obs"]["real_env_steps"]

        self.units = obs["obs"]["units"][self.me]
        self.his_units = obs["obs"]["units"][self.him]

        self.factories = obs["obs"]["factories"][self.me]
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

            if "valid_spawns_mask" in self.board:
                #print(self.board["valid_spawns_mask"], file=sys.stderr)
                self.valid_spawns_mask = np.array(self.board["valid_spawns_mask"])

        # obs global_id: 0 ?

        self.bonus_time_left = obs["remainingOverageTime"]
        self.reward = obs["reward"]  #?

