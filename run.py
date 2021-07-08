import gym, ray
from gym import spaces
import numpy as np
from scipy.spatial import distance
# import pdb
import MultiAgentEnv as ma_env

from policy import PolicyNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog

# env = ma_env.MA_env(config={"atoms":["C", "H", "H", "C", "H"]})
# print(env.reset(methane))
# # observation, reward = env.reset()
# # random_disp = action_space_domain(observation)
# step_to_take = np.random.uniform(-0.5,0.5,(5,3))
# new_state, reward, done,info = env.step(step_to_take)
# print(new_state, reward, done,info)


# create NN model for each atom type
model_C = PolicyNetwork
model_H = PolicyNetwork
ModelCatalog.register_custom_model("modelC", model_C)
ModelCatalog.register_custom_model("modelH", model_H)

# define action space and observation space
# action space is step the policy takes in angstrom
# observation space are the coordinates of the single atom
act_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
obs_space = spaces.Box(low=-10000,high=10000, shape=(768,))

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


from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
import os
from ray.tune.logger import pretty_print

def env_creator(env_config):
    return ma_env.MA_env(env_config)  # return an env instance

register_env("MA_env", env_creator)

config = ppo.DEFAULT_CONFIG.copy()

config["multiagent"] = {
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policies,
        "policies_to_train": ["policy_C", "policy_H"],
        "count_steps_by": "env_steps"
    }

config["log_level"] = "WARN"
config["framework"] = "torch"
config["num_gpus"] =  int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["env_config"] =  {"atoms":["C", "H", "H", "H", "H"]}
config["rollout_fragment_length"] = 16
config["sgd_minibatch_size"] = 16
config["train_batch_size"] = 160
config["num_workers"] = 4
# config["monitor"] = True

print(pretty_print(config))

# quit()
ray.init()
agent = ppo.PPOTrainer(config, env="MA_env")

n_iter = 200
for n in range(n_iter):
    result = agent.train()
    print(pretty_print(result))

    if n % 10 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)
