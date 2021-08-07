from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton, BFGS
from gpaw import GPAW, PW, FD
from ase.calculators.gaussian import Gaussian
from ase.io.trajectory import Trajectory
import pdb
import numpy as np
import matplotlib.pyplot as plt

# system = Atoms('CH4', positions=[[-0.02209687,  0.00321505,  0.01651974],
#                                 [-0.66900878,  0.88935986, -0.1009085 ],
#                                 [-0.37778794, -0.85775189, -0.58829603],
#                                 [ 0.09642092, -0.3151253 ,  1.06378087],
#                                 [ 0.97247267,  0.28030227, -0.39109608]])

#non-eq methane
system = Atoms('CH4', positions=[[-1.65678048,    0.70894727,    0.28577386],
                                 [-1.32345858,   -0.23386581,    0.28577386],
                                 [-1.39010920,    1.08606742,    0.93897124],
                                 [-1.15677183,    1.41604753,   -0.93897124],
                                 [-3.25678048,    0.70896698,    0.28577386]])
# pdb.set_trace()
# system = Atoms("Ga31", positions=[[16.25832, 11.45795, 14.78544],
#                                   [12.88887, 15.24319, 10.21792],
#                                   [10.70294, 13.19028,  9.96131],
#                                   [13.27126, 12.54014, 11.05947],
#                                   [10.85114, 15.34559, 11.93199],
#                                   [13.66015, 12.07448, 14.02008],
#                                   [14.00908, 14.20350, 15.85968],
#                                   [15.07433,  9.54680, 13.22886],
#                                   [ 9.24024, 12.76075, 15.26340],
#                                   [11.95710, 10.47752,  9.57490],
#                                   [13.10528,  9.58000, 15.25256],
#                                   [14.00903, 10.69664,  7.76248],
#                                   [12.47401, 10.23303, 12.44864],
#                                   [10.55330, 10.21642, 14.49926],
#                                   [16.03587, 11.87017, 12.02115],
#                                   [15.47692, 12.33075,  9.42385],
#                                   [ 8.22574, 11.00383, 13.32476],
#                                   [15.95646, 14.00154, 13.82351],
#                                   [ 8.79692, 13.33011, 11.65983],
#                                   [11.57834, 12.19260, 16.20373],
#                                   [15.44697, 14.50788, 11.13911],
#                                   [13.32155, 14.55191, 13.02390],
#                                   [14.55432,  9.95787, 10.52870],
#                                   [ 7.04052, 13.40453, 13.66419],
#                                   [ 9.95093, 10.70643, 11.31321],
#                                   [14.07814, 11.62404, 16.83094],
#                                   [14.79691, 14.75751,  8.30437],
#                                   [11.43757, 14.76123, 15.09893],
#                                   [12.65310, 13.03897,  8.24829],
#                                   [11.05243, 12.69840, 13.13754],
#                                   [ 9.04228, 15.19593, 13.88800]])

# calc = EMT()
# system.calc = calc
# print(f"{system.get_potential_energy() * 23.0605} Kcal/mol")
# print(f"{system.get_potential_energy()} eV")
# print(system.force())

# Gaussian

# system.calc = Gaussian(label='calc/gaussian', 
#                         xc='B3LYP',
#                         basis='6-31+G*',
#                         scf=['qc', 'maxcycle=300'])
# print(f"{system.get_potential_energy()} Kcal/mol")
# print(f"{system.get_potential_energy()} eV")
# print(system.force())

# pdb.set_trace()
system.set_cell((20.0, 20.0, 20.0))
system.center(vacuum=3.0)
# calc = GPAW(mode='lcao', basis='dzp', txt='gpaw.txt')
# calc = EMT()

# calc = GPAW(xc="PBE", mode=PW(300), txt="ga31.txt")
calc = GPAW(xc="PBE", mode=FD(nn=3), txt="methane_fd_pbe.txt")
system.calc = calc
print(f"{system.get_potential_energy()} eV")
print(system.get_forces())
print(system.get_all_distances())

dyn = BFGS(system, trajectory='methane.traj')
smvar = dyn.run(fmax=0.05)
print(f"{system.get_potential_energy()} eV")
print(system.get_forces())
print(system.get_all_distances())
print(smvar)
# from aev import AEVComputer

# Rcr = 5.2000e+00
# Rca = 3.5000e+00

# EtaR = torch.tensor([1.6000000e+01], device=device)
# ShfR = torch.tensor([9.0000000e-01, 1.0193548e+00, 1.1387097e+00, 1.2580645e+00, 1.3774194e+00, 1.4967742e+00, 1.6161290e+00, 1.7354839e+00, 1.8548387e+00, 1.9741935e+00, 2.0935484e+00, 2.2129032e+00, 2.3322581e+00, 2.4516129e+00, 2.5709677e+00, 2.6903226e+00, 2.8096774e+00, 2.9290323e+00, 3.0483871e+00, 3.1677419e+00, 3.2870968e+00, 3.4064516e+00, 3.5258065e+00, 3.6451613e+00, 3.7645161e+00, 3.883871e+00, 4.0032258e+00, 4.1225806e+00, 4.2419355e+00, 4.3612903e+00, 4.4806452e+00, 4.6e+00], device=device)
# Zeta = torch.tensor([3.2000000e+01], device=device)
# ShfZ = torch.tensor([1.9634954e-01, 5.8904862e-01, 9.8174770e-01, 1.3744468e+00, 1.7671459e+00, 2.1598449e+00, 2.5525440e+00, 2.9452431e+00], device=device)
# EtaA = torch.tensor([8.0000000e+00], device=device)
# ShfA = torch.tensor([9.0000000e-01, 1.1785714e+00, 1.4571429e+00, 1.7357143e+00, 2.0142857e+00, 2.2928571e+00, 2.5714286e+00, 2.8500000e+00], device=device)
# num_species = 4

# aev_computer = torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
