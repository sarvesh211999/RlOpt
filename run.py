import gym, ray
from gym import spaces
import numpy as np
from scipy.spatial import distance
import pdb
import MultiAgentEnv as ma_env

from policy import PolicyNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray import tune
import ray.rllib.agents.ppo as ppo
import os
from ray.tune.logger import pretty_print
from ray.tune.logger import Logger


from typing import Dict
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

from datetime import datetime

LOG_FILE = "/home/rohit/ssds/nnp/proj7_RL4Opt/RlOpt/logs/log_file_{}.txt".format(datetime.now().strftime("%d_%m_%H_%M"))


class MyCallbacks(DefaultCallbacks):
    # def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
    #                      policies: Dict[str, Policy],
    #                      episode: MultiAgentEpisode, env_index: int, **kwargs):
    #     # Make sure this episode has just been started (only initial obs
    #     # logged so far).
    #     assert episode.length == 0, \
    #         "ERROR: `on_episode_start()` callback should be called right " \
    #         "after env reset!"
    #     print("episode {} (env-idx={}) started.".format(
    #         episode.episode_id, env_index))

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        # print(episode.last_info_for("C_1"))
        # # Make sure this episode is really done.
        # print("episode {} (env-idx={}) end.".format(
        #     episode.episode_id, env_index))
        # print("----------------------------------")
        # pdb.set_trace()
        trajectories =  base_env.get_unwrapped()[0].trajectory
        energies =  base_env.get_unwrapped()[0].energies

        # print(trajectories)
        # print(energies)

        with open(LOG_FILE, "a") as outFile:
            for idx,trajectory in enumerate(trajectories):
                outFile.write("episode_id: {} \t env-idx: {} \t pos:{} \t energy:{}\n".format(episode.episode_id, env_index, str(list(trajectory.flatten())),energies[idx]))


    # def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
    #                   **kwargs):
    #     print("returned sample batch of size {}".format(samples.count))

    # def on_train_result(self, *, trainer, result: dict, **kwargs):
    #     print("trainer.train() result: {} -> {} episodes".format(
    #         trainer, result["episodes_this_iter"]))
    #     # you can mutate the result dict to add new fields to return
    #     result["callback_ok"] = True

    # def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
    #                       result: dict, **kwargs) -> None:
    #     result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
    #     print("policy.learn_on_batch() result: {} -> sum actions: {}".format(
    #         policy, result["sum_actions_in_train_batch"]))

    # def on_postprocess_trajectory(
    #         self, *, worker: RolloutWorker, episode: MultiAgentEpisode,
    #         agent_id: str, policy_id: str, policies: Dict[str, Policy],
    #         postprocessed_batch: SampleBatch,
    #         original_batches: Dict[str, SampleBatch], **kwargs):
    #     print("postprocessed {} steps".format(postprocessed_batch.count))
    #     if "num_batches" not in episode.custom_metrics:
    #         episode.custom_metrics["num_batches"] = 0
    #     episode.custom_metrics["num_batches"] += 1


# create NN model for each atom type
model_C = PolicyNetwork
model_H = PolicyNetwork
ModelCatalog.register_custom_model("modelC", model_C)
ModelCatalog.register_custom_model("modelH", model_H)

# define action space and observation space
# action space is step the policy takes in angstrom
# observation space are the coordinates of the single atom
act_space = spaces.Box(low=-0.1,high=0.1, shape=(3,))
obs_space = spaces.Box(low=-10000,high=10000, shape=(768,))

def gen_policy(atom):
    model = "model{}".format(atom)
    config = {"model": {"custom_model": model,},}
    return (None, obs_space, act_space, config)

policies = {"policy_C": gen_policy("C"),"policy_H": gen_policy("H")}
policy_ids = list(policies.keys())

def policy_mapping_fn(agent_id, episode, **kwargs):
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
config["rollout_fragment_length"] = 32
config["sgd_minibatch_size"] = 16
config["train_batch_size"] = 96
config["num_workers"] = 4
config["callbacks"] = MyCallbacks
# config["record_env"] = True

print(pretty_print(config))

ray.init()
agent = ppo.PPOTrainer(config, env="MA_env")

n_iter = 200
for n in range(n_iter):
    result = agent.train()
    print(pretty_print(result))

    if n % 5 == 0:
        checkpoint = agent.save()
        print("checkpoint saved at", checkpoint)