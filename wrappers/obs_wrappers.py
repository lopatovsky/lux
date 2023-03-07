from typing import Any, Dict

import gym
import numpy as np
import numpy.typing as npt
from gym import spaces

from game_state import GameState

class LuxObservationWrapper(gym.ObservationWrapper):
    """
    Observation wrapper for Lux AI 2 environment.
    """

    observation_features = 17

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            -999, 999, shape=(LuxObservationWrapper.observation_features,))

    def observation(self, obs):
        return LuxObservationWrapper.convert_obs(obs, self.env.state.env_cfg)

    # we make this method static so the submission/evaluation code can use this as well
    @staticmethod
    def convert_obs(state: GameState) -> Dict[str, npt.NDArray]:
        env_cfg = state.cfg

        ice_map = state.ice
        # TODO from game_state greedily or preprocess
        ice_tile_locations = np.argwhere(ice_map == 1)

        ore_map = state.ore
        ore_tile_locations = np.argwhere(ore_map == 1)


        obs_vec = np.zeros(
            LuxObservationWrapper.observation_features,
        )

        factories = state.factories
        factory_vec = np.zeros(2)
        factory_cargo = np.zeros(2)
        for k in factories.keys():
            # here we track a normalized position of the first friendly factory
            factory = factories[k]
            factory_cargo = np.array([
                factory["cargo"]["water"] / 1000,
                factory["cargo"]["metal"] / 1000
            ])
            factory_vec = np.array(factory["pos"]) / env_cfg.map_size
            break
        units = state.units
        for k in units.keys():
            unit = units[k]

            # store cargo+power values scaled to [0, 1]
            cargo_space = env_cfg.ROBOTS[unit["unit_type"]].CARGO_SPACE
            battery_cap = env_cfg.ROBOTS[unit["unit_type"]].BATTERY_CAPACITY
            cargo_vec = np.array(
                [
                    unit["power"] / battery_cap,
                    unit["cargo"]["ice"] / cargo_space,
                    unit["cargo"]["ore"] / cargo_space,
                    unit["cargo"]["water"] / cargo_space,
                    unit["cargo"]["metal"] / cargo_space,
                ]
            )
            unit_type = (
                0 if unit["unit_type"] == "LIGHT" else 1
            )  # note that build actions use 0 to encode Light
            # normalize the unit position
            pos = np.array(unit["pos"]) / env_cfg.map_size
            unit_vec = np.concatenate(
                [pos, [unit_type], cargo_vec, [unit["team_id"]]], axis=-1
            )

            # we add some engineered features down here
            # compute closest ice tile
            ice_tile_distances = np.mean(
                (ice_tile_locations - np.array(unit["pos"])) ** 2, 1
            )
            # normalize the ice tile location
            closest_ice_tile = (
                ice_tile_locations[np.argmin(ice_tile_distances)] / env_cfg.map_size
            )

            # compute closest ore tile
            ore_tile_distances = np.mean(
                (ore_tile_locations - np.array(unit["pos"])) ** 2, 1
            )
            # normalize the ore tile location
            closest_ore_tile = (
                    ore_tile_locations[np.argmin(ore_tile_distances)] / env_cfg.map_size
            )


            obs_vec = np.concatenate(
                [unit_vec, factory_vec - pos, closest_ice_tile - pos, closest_ore_tile - pos, factory_cargo ], axis=-1
            )
            break


        return obs_vec
