import pickle

import cloudpickle
import gym
import numpy as np
import torch


def load(path):
    with open(f"{path}", "rb") as f:
        obj = pickle.load(f)
    return obj


def save(
    args,
    n_evals,
    b_evals,
    a_evals,
    archive,
    greedy,
    env_rng_states,
    kdt,
    count,
):
    checkpoint_dict = {
        "n_evals": n_evals,
        "b_evals": b_evals,
        "a_evals": a_evals,
        "archive": archive,
        "kdt": kdt,
        "greedy": greedy,
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "env_rng_states": env_rng_states,
        "counter": count,
    }
    with open(f"{args.save_path}/checkpoint/{args.file_name}_checkpoint", "wb") as f:
        pickle.dump(checkpoint_dict, f)


def make_env(env_id, random_state=None):
    env = gym.make(env_id)
    if random_state:
        env.np_random.set_state(random_state)
        env.robot.np_random.set_state(random_state)
    return env


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_env.py#L190
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        self.x = pickle.loads(ob)
