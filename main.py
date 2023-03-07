import json
import os.path as osp
from stable_baselines3.ppo import PPO

from agent import Agent
from game_state import GameState

def read_input():
    """
    Reads input from stdin
    """
    try:
        return input()
    except EOFError as eof:
        raise SystemExit(eof)

def load_model(path: str):
    #directory = osp.dirname(__file__)
    #return PPO.load(osp.join(directory, MODEL_WEIGHTS_RELATIVE_PATH))
    return None # TODO __file__ is failing

MODEL_WEIGHTS_RELATIVE_PATH = "./best_model"


if __name__ == "__main__":


    ppo_model = load_model(MODEL_WEIGHTS_RELATIVE_PATH)
    state = None
    agent = None

    while True:
        inputs = json.loads(read_input())

        if agent == None:
            state = GameState(inputs)
            agent = Agent(ppo_model, state)
        else:
            state.update(inputs)

        actions = agent.step()

        # send actions to engine
        print(json.dumps(actions))
