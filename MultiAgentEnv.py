import gym, ray, pdb
from gym import spaces
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
# cartesian x, y, z. if in spherical coordinates r, theta, phi
# action_space = [x_dir, y_dir, z_dir]

# observation_space = [coordinates, delta_coor, force, energy]

# zs = ["C", "H", "H", "H", "H"]

methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
                   [-0.66900878,  0.88935986, -0.1009085 ],
                   [-0.37778794, -0.85775189, -0.58829603],
                   [ 0.09642092, -0.3151253 ,  1.06378087],
                   [ 0.97247267,  0.28030227, -0.39109608]])

class MA_env(MultiAgentEnv):
    def __init__(self, env_config):
        # pick actual env based on worker and env indexes
        self.dones = set()
        self.agents = []
        self.atom_count = {}
        self.atom_agent_map = []
        for val in env_config["atoms"]:
            if val not in self.atom_count:
                self.atom_count[val] = 0
            self.atom_count[val] += 1
            self.atom_agent_map.append(val+"_"+str(self.atom_count[val]))
        for key in env_config["atoms"]:
            self.agents.append(Atom_Agent())

        self.action_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
        self.observation_space = spaces.Box(low=-100,high=100, shape=(3,))
    
    def reset(self):
        self.dones = set()

        ## set new molecule from dataset here
        coordinates = methane
        return_dict = {self.atom_agent_map[i]: agent.reset(coordinates[i]) for i, agent in enumerate(self.agents)}

        return return_dict

    def step(self, action):
        obs, rew, done, info = {}, {}, {}, {}
        for val in self.atom_agent_map:
            obs[val] = None
            rew[val] = None
            done[val] = None
            info[val] = None

        for idx, val in action.items():
            
            obs[idx], rew[idx], done[idx], info[idx] = self.agents[self.atom_agent_map.index(idx)].step(val)
            if done[idx]:
                self.dones.add(idx)
        done["__all__"] = len(self.dones) == len(self.agents)

        return obs, rew, done, info


class Atom_Agent(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
        self.observation_space = spaces.Box(low=-100,high=100, shape=(3,))

    def reset(self,coordinates):
        self.coordinates = coordinates

        ## return features instead of coordintaes
        return self.coordinates
    
    def step(self, action):

        ## return obs, rew, terminmated,info
        self.coordinates = np.array(self.coordinates) + np.array(action)
        
        ## return features instead of coordintaes
        ## return coordinates in info dict last dict
        terminate = False
        if np.any(self.coordinates) > 100 or np.any(self.coordinates) < -100:
            terminate = True
        return self.coordinates, 1,terminate, {}
