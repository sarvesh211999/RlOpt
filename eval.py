#!/usr/bin/env python
# coding: utf-8

# In[28]:


import matplotlib
import numpy as np
import matplotlib.pyplot as plt

import gym, ray
from gym import spaces
from scipy.spatial import distance
# import pdb
import MultiAgentEnv as ma_env
import tempfile
from ase import Atoms
from gpaw import GPAW, PW, FD
from ase.optimize import QuasiNewton, BFGS
from ase.io.trajectory import Trajectory

from policy import PolicyNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
import os, pdb
from ray.tune.logger import pretty_print
from ray.tune.logger import Logger, UnifiedLogger

from typing import Dict
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from datetime import datetime


# In[2]:


model_C = PolicyNetwork
model_H = PolicyNetwork
ModelCatalog.register_custom_model("modelC", model_C)
ModelCatalog.register_custom_model("modelH", model_H)

act_space = spaces.Box(low=-0.05, high=0.05, shape=(3,))
obs_space = spaces.Box(low=-10000, high=10000, shape=(768,))

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

def policy_mapping_fn(agent_id,  **kwargs):
    if agent_id.startswith("C"):
        pol_id = "policy_C"
    else:
        pol_id = "policy_H"
    return pol_id

def env_creator(env_config):
    return ma_env.MA_env(env_config)  # return an env instance

register_env("MA_env", env_creator)

config = ppo.DEFAULT_CONFIG.copy()

config["multiagent"] = {
        "policy_mapping_fn": policy_mapping_fn,
        "policies": policies,
        "policies_to_train": ["policy_C", "policy_H"],
    }

config["log_level"] = "WARN"
config["framework"] = "torch"
config["num_gpus"] =  int(os.environ.get("RLLIB_NUM_GPUS", "0"))
config["env_config"] =  {"atoms":["C", "H", "H", "H", "H"]}
config["rollout_fragment_length"] = 16
config["in_evaluation"] = True


# In[3]:


def custom_log_creator(custom_path, custom_str):

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = "{}_{}".format(custom_str, timestr)

    def logger_creator(config):

        if not os.path.exists(custom_path):
            os.makedirs(custom_path)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=custom_path)
        return UnifiedLogger(config, logdir, loggers=None)

    return logger_creator


# In[4]:


ray.init()


# In[5]:


agent = ppo.PPOTrainer(config=config, env="MA_env", logger_creator=custom_log_creator(os.path.expanduser("/home/rohit/ssds/nnp/proj7_RL4Opt/RlOpt/results"), 'custom_dir'))


# In[6]:


agent.restore("/home/rohit/ray_results/PPO_MA_env_2021-08-10_15-43-05_discrete_reward/checkpoint_000091/checkpoint-91")


# In[7]:


env = ma_env.MA_env({})


# In[8]:


obs = env.reset()


# In[9]:


# agent.compute_action(obs,full_fetch=True)


# In[10]:


action = {}
# all_info = []
done = False
for i in range(10):
# while done != True:
#     pdb.set_trace()
    for agent_id, agent_obs in obs.items():
    #     print(agent_id, agent_obs)
        policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
    #     action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id)
        action[agent_id] = agent.compute_single_action(agent_obs, policy_id=policy_id)
    obs, rew, done, info = env.step(action)
    print(f"actions: {action}")
    print("\n")
#     all_info.append(info)


# In[11]:


rl_energy = np.array(env.energies)
ray.shutdown()

# In[20]:

# methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
#                    [-0.66900878,  0.88935986, -0.1009085 ],
#                    [-0.37778794, -0.85775189, -0.58829603],
#                    [ 0.09642092, -0.3151253 ,  1.06378087],
#                    [ 0.97247267,  0.28030227, -0.39109608]])


atoms = Atoms('C1H4', positions=np.array([[-0.02209687,  0.00321505,  0.01651974],
                                          [-0.66900878,  0.88935986, -0.1009085 ],
                                          [-0.37778794, -0.85775189, -0.58829603],
                                          [ 0.09642092, -0.3151253 ,  1.06378087],
                                          [ 0.97247267,  0.28030227, -0.39109608]]))

# atoms = Atoms('C1H4', positions=np.array([[-1.65678048,    0.70894727,    0.28577386],
#                                           [-1.32345858,   -0.23386581,    0.28577386],
#                                           [-1.39010920,    1.08606742,    0.93897124],
#                                           [-1.15677183,    1.41604753,   -0.93897124],
#                                           [-3.25678048,    0.70896698,    0.28577386]]))


# ethane = Atoms('CHHHCHHH', positions=np.array([[-1.75355881,   -0.30653455,    0.12937570],
#                                               [-1.33829924,   -1.50668062,    0.01894070],
#                                               [-2.82350094,   -0.29939047,    0.03790805],
#                                               [-1.40045188,    0.20066408,   -0.84410300],
#                                               [-1.22539515,    0.40849176,    1.28690524],
#                                               [-1.57525951,    1.41963846,    1.29569693],
#                                               [-1.55540163,   -0.06552565,    2.10324048],
#                                               [-0.15545302,    0.40134768,    1.27837290]]))

# In[22]:


def do_optim(system, traj_file):
    system.set_cell((20.0, 20.0, 20.0))
    system.center(vacuum=3.0)
    calc = GPAW(xc="PBE", mode=FD(nn=3))
    system.calc = calc

    # print(f"{system.get_potential_energy()} eV")
    # print(system.get_forces())
    # print(system.get_all_distances())

    dyn = BFGS(system, trajectory=traj_file)
    smvar = dyn.run(fmax=0.05)

    # print(f"{system.get_potential_energy()} eV")
    # print(system.get_forces())
    # print(system.get_all_distances())
    # print(smvar)

def read_traj(in_traj):
    traj = Trajectory(in_traj)
    return traj


# In[29]:


# do_optim(atoms, "traj_dir/methane_eval.traj")
methane_bfgs_traj = read_traj("traj_dir/methane_eval.traj")


# In[41]:


bfgs_energy = []
for frame in methane_bfgs_traj:
    bfgs_energy.append(frame.get_potential_energy())
    print(frame.get_potential_energy())
    # print(frame.get_forces())
bfgs_energy = np.array(bfgs_energy)


# In[ ]:


plt.title("Energy vs number of steps")
plt.xlabel("Steps")
plt.ylabel("Energy (eV)")
plt.plot(rl_energy, 'o-', label="RL")
plt.plot(bfgs_energy, 'o-', label="BFGS")
plt.legend()
plt.savefig("optim.png", bbox_inches='tight', dpi=300)

