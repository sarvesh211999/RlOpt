""" original contribution from Andrew Dalke """
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import pdb

writer = Chem.SDWriter("decane_rdk_optim.sdf")

# This function is called in the subprocess.
# The parameters (molecule and number of conformers) are passed via a Python
def generateconformations(m, n):
    m = Chem.AddHs(m)
    ids=AllChem.EmbedMultipleConfs(m, numConfs=n, params=AllChem.ETKDG())
    # EmbedMultipleConfs returns a Boost-wrapped type which
    # cannot be pickled. Convert it to a Python list, which can.
    return m, list(ids)

# m = Chem.MolFromSmiles('CC1=C(C(CCC1)(C)C)C=CC(=CC=CC(=CCO)C)C') #retinol
# m = Chem.MolFromSmiles('CCC(=O)N(C1CCN(CCC2=CC=CC=C2)CC1)C1=CC=CC=C1') #fentanyl
m = Chem.MolFromSmiles('CCCCCCCCCC') #decane
# m = Chem.MolFromSmiles('CC(CC1=CC=CC=C1)NC') #meth
n = 8

mol, ids = generateconformations(m, n)
copy_mol, copy_ids = mol, ids
res = AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, maxIters=600)
print(res)

# pdb.set_trace()
for id in ids:
	writer.write(mol, confId=id)
