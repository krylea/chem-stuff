from ocpmodels.preprocessing import AtomsToGraphs
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
import ase.io
from ase.build import bulk
from ase.build import fcc100, add_adsorbate, molecule
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS
import matplotlib.pyplot as plt
import lmdb
import pickle
from tqdm import tqdm
import torch
import os

#from ocdata.vasp import xml_to_traj

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("run_folders", type=str, nargs='+')
args = parser.parse_args()

a2g = AtomsToGraphs(
    max_neigh=50,
    radius=6,
    r_energy=True,    # False for test data
    r_forces=True,
    r_distances=False,
    r_fixed=True,
)
db = lmdb.open(
    "chem-data.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

def read_trajectory_extract_features(a2g, traj_path, xml=False):
    traj = xml_to_traj(traj_path) if xml else ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects

idx=0
for run_folder in args.run_folders:
    assert os.path.exists(os.path.join(run_folder, "surface"))
    subfolders = [ f.name for f in os.scandir(run_folder) if f.is_dir() if f.name != "surface"]

    data_objects = read_trajectory_extract_features(a2g, os.path.join(run_folder, "surface", "OUTCAR"))
    surface_energy = data_objects[1].y_relaxed
    for runname in subfolders:
            # Extract Data object
        filename = os.path.join(run_folder, runname, "OUTCAR")
        data_objects = read_trajectory_extract_features(a2g, filename)
        initial_struc = data_objects[0]
        relaxed_struc = data_objects[1]
        
        initial_struc.y_init = initial_struc.y - surface_energy # subtract off reference energy, if applicable
        del initial_struc.y
        initial_struc.y_relaxed = relaxed_struc.y - surface_energy # subtract off reference energy, if applicable
        initial_struc.pos_relaxed = relaxed_struc.pos
        
        # Filter data if necessary
        # OCP filters adsorption energies > |10| eV
        
        initial_struc.sid = idx  # arbitrary unique identifier 
        
        # no neighbor edge case check
        if initial_struc.edge_index.shape[1] == 0:
            print("no neighbors", filename)
            continue
        
        # Write to LMDB
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1

db.close()
