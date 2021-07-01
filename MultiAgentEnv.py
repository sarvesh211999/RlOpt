import gym, ray, pdb
from gym import spaces
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
import torchani

from ase import Atoms
from ase.io import Trajectory
from gpaw import GPAW
# cartesian x, y, z. if in spherical coordinates r, theta, phi
# action_space = [x_dir, y_dir, z_dir]

# observation_space = [coordinates, delta_coor, force, energy]

# zs = ["C", "H", "H", "H", "H"]

methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
                   [-0.66900878,  0.88935986, -0.1009085 ],
                   [-0.37778794, -0.85775189, -0.58829603],
                   [ 0.09642092, -0.3151253 ,  1.06378087],
                   [ 0.97247267,  0.28030227, -0.39109608]])
## AEV parameters
Rcr = 5.2000e+00
Rca = 3.5000e+00

EtaR = torch.tensor([1.6000000e+01], device=device)
ShfR = torch.tensor([9.0000000e-01, 1.0193548e+00, 1.1387097e+00, 1.2580645e+00, 1.3774194e+00, 1.4967742e+00, 1.6161290e+00, 1.7354839e+00, 1.8548387e+00, 1.9741935e+00, 2.0935484e+00, 2.2129032e+00, 2.3322581e+00, 2.4516129e+00, 2.5709677e+00, 2.6903226e+00, 2.8096774e+00, 2.9290323e+00, 3.0483871e+00, 3.1677419e+00, 3.2870968e+00, 3.4064516e+00, 3.5258065e+00, 3.6451613e+00, 3.7645161e+00, 3.883871e+00, 4.0032258e+00, 4.1225806e+00, 4.2419355e+00, 4.3612903e+00, 4.4806452e+00, 4.6e+00], device=device)
Zeta = torch.tensor([3.2000000e+01], device=device)
ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
EtaA = torch.tensor([8.0000000e+00], device=device)
ShfA = torch.tensor([9.0000000e-01, 1.1785714e+00, 1.4571429e+00, 1.7357143e+00, 2.0142857e+00, 2.2928571e+00, 2.5714286e+00, 2.8500000e+00], device=device)
num_species = 4

aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
species_to_tensor = torchani.utils.ChemicalSymbolsToInts('HCNO')


def cartesian_to_spherical(pos: np.ndarray) -> np.ndarray:
    theta_phi = np.empty(shape=pos.shape[:-1] + (3, ))

    x, y, z = pos[..., 0], pos[..., 1], pos[..., 2]
    r = np.linalg.norm(pos, axis=-1)
    theta_phi[..., 0] = r
    theta_phi[..., 1] = np.arccos(z / r)  # theta
    theta_phi[..., 2] = np.arctan2(y, x)  # phi

    return theta_phi

def spherical_to_cartesian(theta_phi: np.ndarray) -> np.ndarray:
    r, theta, phi = theta_phi[..., 0], theta_phi[..., 1], theta_phi[..., 2]
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


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
        self.observation_space = spaces.Box(low=-10000,high=10000, shape=(3,))
    
    def reset(self):
        print("Reset called")
        self.dones = set()

        ## set new molecule from dataset here
        # coordinates = cartesian_to_spherical(methane)
        # return_dict = {self.atom_agent_map[i]: agent.reset(coordinates[i]) for i, agent in enumerate(self.agents)}

        """feature vector.
        input can be a tuple of two tensors: species, coordinates.
        species must have shape ``(C, A)``, coordinates must have shape
        ``(C, A, 3)`` where ``C`` is the number of molecules in a chunk,
        and ``A`` is the number of atoms."""
        species = species_to_tensor([["C", "H", "H", "H", "H"]]).unsqueeze(dim=0)
        coor = torch.tensor(methane).unsqueeze(dim=0)
        result = aev_computer((species.to(device), coor.to(device)))
        aev = result.aev
        return_dict = {self.atom_agent_map[i]: agent.reset(aev[0,i]) for i, agent in enumerate(self.agents)}
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

        final_coordinates = np.array([obs[key] for key in self.atom_agent_map])
        cart_coordinates = spherical_to_cartesian(final_coordinates)
        atoms = Atoms('C1H4', positions=cart_coordinates)
        atoms.center(vacuum=3.0)
        try:
            calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt')
            atoms.calc = calc

            f = atoms.get_forces()

            spherical_forces = cartesian_to_spherical(f)

            for idx,key in enumerate(self.atom_agent_map):
                rew[key] = np.abs(1/(spherical_forces[idx][0]))
                if spherical_forces[idx][0] < 2.571103:
                    self.dones.add(idx)
            # print(len(self.dones))
            print(f"Forces: {spherical_forces[:,0]}, opt_atm:{len(self.dones)}")
            print(f"Reward:{rew}")
        except:
            # print("GPAW Converge error")
            for idx, key in enumerate(self.atom_agent_map):
                rew[key] = -1
                self.dones.add(idx)

        ## convert obs to feature vector here

        ## calculate atom wise reward here

        done["__all__"] = len(self.dones) == len(self.agents)

        return obs, rew, done, info


class Atom_Agent(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
        self.observation_space = spaces.Box(low=-10000,high=10000, shape=(3,))

    def reset(self,coordinates):
        self.coordinates = np.array(coordinates)

        ## return features instead of coordintaes
        return self.coordinates
    
    def step(self, action):
        new_coordinates = spherical_to_cartesian(self.coordinates) + np.array(action)
        self.coordinates = cartesian_to_spherical(new_coordinates)

        return self.coordinates, None,False, {}