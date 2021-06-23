import gym, ray
from gym import spaces
import numpy as np
from scipy.spatial import distance
# import pdb
import MultiAgentEnv as ma_env

from policy import PolicyNetwork
from ray.rllib.models import ModelCatalog

# cartesian x, y, z. if in spherical coordinates r, theta, phi
def action_space_domain(coordinates):
    # pdb.set_trace()
    llim, ulim = -0.5, 0.5
    action_space_wdirection = {"x":[llim, ulim], "y":[llim, ulim], "z":[llim, ulim]}
    random_displacment = np.random.uniform(-0.5, 0.5, coordinates.shape)
    return random_displacment
# observation_space = [coordinates, delta_coor, force, energy]

methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
                   [-0.66900878,  0.88935986, -0.1009085 ],
                   [-0.37778794, -0.85775189, -0.58829603],
                   [ 0.09642092, -0.3151253 ,  1.06378087],
                   [ 0.97247267,  0.28030227, -0.39109608]])


# pdb.set_trace()
# env = ma_env.MA_env(config={"atoms":["C", "H", "H", "C", "H"]})
# print(env.reset(methane))
# # observation, reward = env.reset()
# # random_disp = action_space_domain(observation)
# step_to_take = np.random.uniform(-0.5,0.5,(5,3))
# new_state, reward, done,info = env.step(step_to_take)
# print(new_state, reward, done,info)



model_C = PolicyNetwork
model_H = PolicyNetwork
ModelCatalog.register_custom_model("modelC", model_C)
ModelCatalog.register_custom_model("modelH", model_H)

act_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
obs_space = spaces.Box(low=-100,high=100, shape=(3,))

def gen_policy(atom):
    model = "model{}".format(atom)
    config = {
        "model": {
            "custom_model": model,
        },
    }
    return (None, obs_space, act_space, config)



policies = {"policy_C": gen_policy("C"),"policy_H": gen_policy("H")}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, **kwargs):
    if agent_id.startswith("C"):
        pol_id = "policy_C"
    else:
        pol_id = "policy_H"
    return pol_id


from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from ray import tune
import os

def env_creator(env_config):
    return ma_env.MA_env(env_config)  # return an env instance

register_env("MA_env", env_creator)

config={
    "multiagent": {
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policies,
        "policies_to_train": ["policy_C", "policy_H"],
    },
    "env":"MA_env",
    "framework": "torch",
    "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
    "env_config": {"atoms":["C", "H", "H", "C", "H"]}
    
}

stop = {
    "episode_reward_mean": 150.0,
    "timesteps_total": 100000,
    "training_iteration": 200,
}
results = tune.run("PPO", stop=stop,config=config, verbose=1)
ray.shutdown()
