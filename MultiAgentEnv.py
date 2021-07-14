import gym, ray, pdb
from gym import spaces
import numpy as np
from ray.tune.registry import register_env
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances

from ase import Atoms
from ase.io import Trajectory
from ase.calculators.emt import EMT
from gpaw import GPAW
import torch
import torchani

# non-eq methane
# energy = -17.76212015884605 eV
# forces = [[-1.27508078e+01 -6.22725271e+00 -2.48573744e+01]
#  [ 1.23733381e+00 -4.76242919e+00 -5.68793880e-01]
#  [ 8.40681290e+00  1.32375926e+01  2.15847319e+01]
#  [-1.74132034e+00 -2.26577915e+00  3.89253669e+00]
#  [ 4.83618538e+00  2.38282237e-02 -6.16713609e-02]]

methane = np.array([[-1.65678048,    0.70894727,    0.28577386],
                    [-1.32345858,   -0.23386581,    0.28577386],
                    [-1.39010920,    1.08606742,    0.93897124],
                    [-1.15677183,    1.41604753,   -0.93897124],
                    [-3.25678048,    0.70896698,    0.28577386]])

# eq methane
# energy = 24.
# methane = np.array([[-0.02209687,  0.00321505,  0.01651974],
#                    [-0.66900878,  0.88935986, -0.1009085 ],
#                    [-0.37778794, -0.85775189, -0.58829603],
#                    [ 0.09642092, -0.3151253 ,  1.06378087],
#                    [ 0.97247267,  0.28030227, -0.39109608]])

bonds = [(0,1),(0,2),(0,3),(0,4)]

## AEV parameters
Rcr = 5.2000e+00
Rca = 3.5000e+00

device=torch.device("cpu")

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
        # pdb.set_trace()
        self.atoms = ["C", "H", "H", "H", "H"]
        self.dones = set()
        self.agents = []
        self.atom_count = {}
        self.atom_agent_map = []
        for val in self.atoms:
            if val not in self.atom_count:
                self.atom_count[val] = 0
            self.atom_count[val] += 1
            self.atom_agent_map.append(val+"_"+str(self.atom_count[val]))
        for key in self.atoms:
            self.agents.append(Atom_Agent())

        self.action_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
        self.observation_space = spaces.Box(low=-10000,high=10000, shape=(768,))

        self.energies = []
    
    def reset(self):
        print("\nReset called")
        # print("Energies:    ",self.energies)
        self.dones = set()
        self.energies = []
        self.curr_coordinates = methane

        ## set new molecule from dataset here
        # coordinates = cartesian_to_spherical(methane)
        # return_dict = {self.atom_agent_map[i]: agent.reset(coordinates[i]) for i, agent in enumerate(self.agents)}

        """feature vector.
        input can be a tuple of two tensors: species, coordinates.
        species must have shape ``(C, A)``, coordinates must have shape
        ``(C, A, 3)`` where ``C`` is the number of molecules in a chunk,
        and ``A`` is the number of atoms."""
        species = species_to_tensor(["C", "H", "H", "H", "H"]).unsqueeze(dim=0)
        coor = torch.tensor(self.curr_coordinates).unsqueeze(dim=0)
        result = aev_computer((species.to(device), coor.to(device)))
        aev = result.aevs
        return_dict = {self.atom_agent_map[i]: agent.reset(aev[0,i],self.curr_coordinates[i]) for i, agent in enumerate(self.agents)}
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

        final_coordinates = np.array([obs[key] for key in self.atom_agent_map])

        species = species_to_tensor(["C", "H", "H", "H", "H"]).unsqueeze(dim=0)
        coor = torch.tensor(final_coordinates).unsqueeze(dim=0)
        result = aev_computer((species.to(device), coor.to(device)))
        aev = result.aevs
        obs = {self.atom_agent_map[i]: np.array(aev[0,i]) for i, agent in enumerate(self.agents)}
        # cart_coordinates = spherical_to_cartesian(final_coordinates)

        atoms = Atoms('C1H4', positions=final_coordinates)
        atoms.center(vacuum=3.0)

        # calculte euclidean distance
        dist_mat = euclidean_distances(atoms.get_positions())
        
        # check if bond breaks with distance cutoff
        bond_dist = np.array([dist_mat[i] for i in bonds]) < 2.0

        if np.all(bond_dist):
            try:
                calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt')
                # calc = EMT()

                atoms.calc = calc

                f = atoms.get_forces()
                e = atoms.get_potential_energy()
                self.energies.append(e)
                
                spherical_forces = cartesian_to_spherical(f)

                for idx,key in enumerate(self.atom_agent_map):
                    rew[key] = np.abs(1/(spherical_forces[idx][0]))
                    # force should be less than 0.01 eV/A 
                    # 1 Hartree/Bohr = 51.42208619083232 eV/A
                    # 2.571103
                    if spherical_forces[idx][0] < 0.01:
                        self.dones.add(idx)
                print(f"forces = {spherical_forces[:,0]} energies = {e} coordinates {atoms.get_positions().flatten()}")
                # print(f"energies = {self.energies}")
                # print(f"bonds = {np.array([dist_mat[i] for i in bonds])}")
            except:
                print("GPAW Converge error")
                print(f"bonds {np.array([dist_mat[i] for i in bonds])}")
                for idx,key in enumerate(self.atom_agent_map):
                    rew[key] = -0.20
                    self.dones.add(idx)
        else:
            print(f"Bond larger that 2.0 A: {np.array([dist_mat[i] for i in bonds])}")
            for idx,key in enumerate(self.atom_agent_map):
                rew[key] = -0.20
                self.dones.add(idx)

        ## convert obs to feature vector here

        ## calculate atom wise reward here

        done["__all__"] = len(self.dones) == len(self.agents)

        return obs, rew, done, info


class Atom_Agent(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(low=-0.5,high=0.5, shape=(3,))
        self.observation_space = spaces.Box(low=-10000,high=10000, shape=(768,))

    def reset(self,feature,coordinates):
        self.feature = np.array(feature)
        self.coordinates = coordinates

        ## return features instead of coordintaes
        return self.feature
    
    def step(self, action):
        self.coordinates = self.coordinates + np.array(action)
        return self.coordinates, None,False, {}
