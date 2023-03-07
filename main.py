import json
import sys
import os.path as osp
from stable_baselines3.ppo import PPO

from argparse import Namespace
from lux.config import EnvConfig

from agent import Agent
from game_state import GameState

### DO NOT REMOVE THE FOLLOWING CODE ###
agent_dict = (
    dict()
)  # store potentially multiple dictionaries as kaggle imports code directly
state: GameState = None

def load_model(path: str):
    #directory = osp.dirname(__file__)
    #return PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))
    return None # TODO __file__ is failing

MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"

def agent_fn(observation, configurations):
    global agent_dict
    global state

    player = observation.player

    observation.obs = json.loads(observation.obs)

    if state is None:
        #print("inpu", file=sys.stderr)
        state = GameState(observation, EnvConfig.from_dict(configurations["env_cfg"]))
        ppo_model = None # TODO populate
        agent_dict[player] = Agent(ppo_model, state)
    else:
        state.update(observation)

    #print(step, state.step, file=sys.stderr)

    agent = agent_dict[player]

    return agent.step()


if __name__ == "__main__":

    def read_input():
        """
        Reads input from stdin
        """
        try:
            return input()
        except EOFError as eof:
            raise SystemExit(eof)

    step = 0
    player_id = 0
    configurations = None
    i = 0
    while True:
        inputs = read_input()
        obs = json.loads(inputs)

        observation = Namespace(
            **dict(
                step=obs["step"],
                obs=json.dumps(obs["obs"]),
                remainingOverageTime=obs["remainingOverageTime"],
                player=obs["player"],
                info=obs["info"],
            )
        )
        if i == 0:
            configurations = obs["info"]["env_cfg"]
        i += 1
        actions = agent_fn(observation, dict(env_cfg=configurations))
        # send actions to engine
        print(json.dumps(actions))
