import sys
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
import random
from gym import spaces

from game_state import GameState


# Controller class copied here since you won't have access to the luxai_s2 package directly on the competition server
class Controller:
    def __init__(self, action_space: spaces.Space) -> None:
        self.action_space = action_space

    def action_to_lux_action(
        self, action: npt.NDArray
    ):
        """
        Takes as input the current "raw observation" and the parameterized action and returns
        an action formatted for the Lux env
        """
        raise NotImplementedError()

    def action_masks(self):
        """
        Generates a boolean action mask indicating in each discrete dimension whether it would be valid or not
        """
        raise NotImplementedError()


class LuxController(Controller):
    def __init__(self, state: GameState) -> None:
        """
        A simple controller that controls only the robot that will get spawned.
        Moreover, it will always try to spawn one heavy robot if there are none regardless of action given

        For the robot unit
        - 4 cardinal direction movement (4 dims)
        - a move center no-op action (1 dim)
        - transfer action just for transferring ice in 4 cardinal directions or center (5)
        - pickup action for power (1 dims)
        - dig action (1 dim)
        - no op action (1 dim) - equivalent to not submitting an action queue which costs power

        It does not include
        - self destruct action
        - recharge action
        - planning (via actions executing multiple times or repeating actions)
        - factory actions
        - transferring power or resources other than ice

        To help understand how to this controller works to map one action space to the original lux action space,
        see how the lux action space is defined in luxai_s2/spaces/action.py

        """
        self.state = state

        self.env_cfg = state.cfg
        self.move_act_dims = 4
        self.transfer_act_dims = 5
        self.pickup_act_dims = 1
        self.dig_act_dims = 1
        self.no_op_dims = 1

        self.move_dim_high = self.move_act_dims
        self.transfer_dim_high = self.move_dim_high + self.transfer_act_dims
        self.pickup_dim_high = self.transfer_dim_high + self.pickup_act_dims
        self.dig_dim_high = self.pickup_dim_high + self.dig_act_dims
        self.no_op_dim_high = self.dig_dim_high + self.no_op_dims

        self.total_act_dims = self.no_op_dim_high
        action_space = spaces.Discrete(self.total_act_dims)
        super().__init__(action_space)

    move_deltas = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])

    def random_move(self, loc, factories):
        move_1 = random.randint(0, 4)
        move_2 = random.randint(0, 4)
        move = 2 * LuxController.move_deltas[move_1] + \
               LuxController.move_deltas[move_2]

        for factory_id in factories.keys():
            factory_loc = factories[factory_id]["pos"]
            dist = np.abs((loc+move) - factory_loc)
            if dist.max() < 2:
                return self.random_move(loc, factories)
        return np.array([[0, move_1, 0, 0, 0, 1], [0, move_1, 0, 0, 0, 1],
                         [0, move_2, 0, 0, 0, 1], self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0), self._get_dig_action(0)])

    def _is_move_action(self, id):
        return id < self.move_dim_high

    def _get_move_action(self, id):
        # move direction is id + 1 since we don't allow move center here
        return np.array([0, id + 1, 0, 0, 0, 1])

    def _is_transfer_action(self, id):
        return id < self.transfer_dim_high

    def _get_transfer_action(self, id):
        id = id - self.move_dim_high
        transfer_dir = id % 5
        return np.array([1, transfer_dir, 0, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_pickup_action(self, id):
        return id < self.pickup_dim_high

    def _get_pickup_action(self, id):
        return np.array([2, 0, 4, self.env_cfg.max_transfer_amount, 0, 1])

    def _is_dig_action(self, id):
        return id < self.dig_dim_high

    def _get_dig_action(self, id):
        return np.array([3, 0, 0, 0, 0, 1])

    def has_space_to_spawn_unit(self, factory, units):
        pos = factory["pos"]
        for unit_id in units.keys():
            if units[unit_id]["pos"][0] == pos[0] and units[unit_id]["pos"][1] == pos[1]:
                return False  # not to spawn new unit if other is present there.
        return True

    def action_to_lux_action(
        self, action: npt.NDArray
    ):
        shared_obs = self.state.obs
        agent = self.state.me

        lux_action = dict()
        units = shared_obs["units"][agent]
        for unit_id in units.keys():
            unit = units[unit_id]
            if unit["unit_type"] == "HEAVY":

                choice = action
                action_queue = []
                no_op = False
                if self._is_move_action(choice):
                    action_queue = [self._get_move_action(choice)]
                elif self._is_transfer_action(choice):
                    action_queue = [self._get_transfer_action(choice)]
                elif self._is_pickup_action(choice):
                    action_queue = [self._get_pickup_action(choice)]
                elif self._is_dig_action(choice):
                    action_queue = [self._get_dig_action(choice)]
                else:
                    # action is a no_op, so we don't update the action queue
                    no_op = True

                # simple trick to help agents conserve power is to avoid updating the action queue
                # if the agent was previously trying to do that particular action already
                if len(unit["action_queue"]) > 0 and len(action_queue) > 0:
                    same_actions = (unit["action_queue"][0] == action_queue[0]).all()
                    if same_actions:
                        no_op = True
                if not no_op:
                    lux_action[unit_id] = action_queue

            else:  # LIGHT unit
                if len(unit["action_queue"]) == 0:
                    lux_action[unit_id] = \
                        self.random_move(unit["pos"], shared_obs["factories"][agent])


        factories = shared_obs["factories"][agent]
        if len(units) == 0:
            for unit_id in factories.keys():
                lux_action[unit_id] = 1  # build a single heavy

        else:
            for factory_id in factories.keys():
                factory = factories[factory_id]
                if factory["cargo"]["metal"] < 10:
                    continue

                if not self.has_space_to_spawn_unit(factory, units):
                    continue

                lux_action[factory_id] = 0

        return lux_action

    def action_masks(self):
        """
        Defines a simplified action mask for this controller's action space

        Doesn't account for whether robot has enough power
        """

        # compute a factory occupancy map that will be useful for checking if a board tile
        # has a factory and which team's factory it is.
        shared_obs = self.state.obs
        agent = self.state.me

        factory_occupancy_map = (
            np.ones_like(shared_obs["board"]["rubble"], dtype=int) * -1
        )
        factories = dict()
        for player in shared_obs["factories"]:
            factories[player] = dict()
            for unit_id in shared_obs["factories"][player]:
                f_data = shared_obs["factories"][player][unit_id]
                f_pos = f_data["pos"]
                # store in a 3x3 space around the factory position it's strain id.
                # factory_occupancy_map[
                #     f_pos[0] - 1 : f_pos[0] + 2, f_pos[1] - 1 : f_pos[1] + 2
                # ] = f_data["strain_id"]

        units = shared_obs["units"][agent]
        action_mask = np.zeros((self.total_act_dims), dtype=bool)
        for unit_id in units.keys():
            action_mask = np.zeros(self.total_act_dims)
            # movement is always valid
            action_mask[:4] = True

            # transferring is valid only if the target exists
            unit = units[unit_id]
            pos = np.array(unit["pos"])
            # a[1] = direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
            move_deltas = LuxController.move_deltas
            for i, move_delta in enumerate(move_deltas):
                transfer_pos = np.array(
                    [pos[0] + move_delta[0], pos[1] + move_delta[1]]
                )

            # dig is valid only if on top of tile with rubble or resources or lichen

            # no-op is always valid
            action_mask[-1] = True
            break
        return action_mask
