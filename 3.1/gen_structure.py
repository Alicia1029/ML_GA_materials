# Get the generated structures from eval_gen.pt file, the CDVAE output

import torch
import numpy as np
from pymatgen.core import Structure, Element, Lattice

data = torch.load("eval_gen.pt")
num_structures = len(data["num_atoms"][0])
num_atoms = data["num_atoms"][0]
chunks = np.cumsum(num_atoms)
frac_coords = data["frac_coords"][0]
atom_types = data["atom_types"][0]

coords = torch.tensor_split(frac_coords, chunks)
atoms = torch.tensor_split(atoms_types, chunks)

angles = data["angles"][0]
lengths = data["lengths"][0]
elements=[]
for i in range(num_structures):
    lattice = Lattice.from_parameters(lengths[i][0],lengths[i][1],lengths[i][2],angles[i][0],angles[i][1],angles[i][2],)
    elements.append([Element.from_Z(x) for x in atoms[i]])
    structure = Structure(lattice, elements[i], coords[i], coords_are_cartesian=False)
    structure.to(fmt="cif", filename=f"structure_{i}.cif")

# frac_coords is fractional coordinates (num_structures * num_atoms, 3)
# y is target property
# edge_index defines the graph ([from_index, to_index], total_num_bonds)
# to_jimages is whether a bond crosses a periodic boundary (total_num_bonds, 3)
# angles is cell angles (num_structures, 3)
# lengths is cell lengths (num_structures, 3)
# num_atoms is number of atoms per structure (num_structures, )
# num_bonds is number of bonds per structure (num_structures, )
# atom_types is atomic number (num_structures * num_atoms, )
# batch is which atom belongs to which structure (num_structures * num_atoms, )