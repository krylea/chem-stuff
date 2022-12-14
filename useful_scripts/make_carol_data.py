from atoms_to_graphs import AtomsToGraphs
#from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset
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



NUM_ADS_ATOMS=1
NUM_SURFACE_ATOMS=24

NUM_ELEMENTS=100

Z_ADS_THRESHOLD = 9.0

ADS_ENERGIES={
    1: -3.477,
    8: -7.204,
    6: -7.282,
    7: -8.083
}

def get_dirs(basedir):
    subfolders = [f.path for f in os.scandir(basedir) if f.is_dir()]
    return subfolders

parser = argparse.ArgumentParser()
parser.add_argument("run_folders", type=str, nargs='+')
parser.add_argument("--ev_bounds", type=float, nargs=2)

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
    "data.lmdb",
    map_size=1099511627776 * 2,
    subdir=False,
    meminit=False,
    map_async=True,
)

def read_trajectory_extract_features(a2g, traj_path):
    traj = ase.io.read(traj_path, ":")
    tags = traj[0].get_tags()
    images = [traj[0], traj[-1]]
    data_objects = a2g.convert_all(images, disable_tqdm=True)
    data_objects[0].tags = torch.LongTensor(tags)
    data_objects[1].tags = torch.LongTensor(tags)
    return data_objects


def get_adsorbate_indices(total_system, surface):
    surface_elements = surface.atomic_numbers.int().bincount(minlength=NUM_ELEMENTS)
    system_elements = total_system.atomic_numbers.int().bincount(minlength=NUM_ELEMENTS)
    ads_elements = system_elements - surface_elements
    indices = torch.arange(total_system.atomic_numbers.size(0))
    ads_indices=[]
    for element in ads_elements.nonzero():
        element_indices = indices[total_system.atomic_numbers==element.item()]
        sorted_indices = sorted(element_indices.tolist(), key=lambda i:total_system.pos[i,2].item(), reverse=True)
        ads_indices += sorted_indices[:ads_elements[element]]
    return ads_indices



def process_surface(surface_dir, idx, ev_bounds=(-5,0)):
    print(surface_dir)
    assert os.path.exists(os.path.join(surface_dir, "surface"))
    subfolders = [ f.name for f in os.scandir(surface_dir) if f.is_dir() if f.name != "surface"]

    try:
        data_objects = read_trajectory_extract_features(a2g, os.path.join(surface_dir, "surface", "OUTCAR"))
    except:
        print("Failed to read surface at", surface_dir)
        return idx
    surface = data_objects[1]
    surface_energy = data_objects[1].y
    for runname in subfolders:
            # Extract Data object
        filename = os.path.join(surface_dir, runname, "OUTCAR")
        if os.path.exists(filename):
            print(runname)
            try:
                data_objects = read_trajectory_extract_features(a2g, filename)
            except Exception as e:
                print(e)
                continue
            initial_struc = data_objects[0]
            relaxed_struc = data_objects[1]

            initial_struc.pos_relaxed = relaxed_struc.pos

            indices = list(range(initial_struc.pos.size(0)))
            ads_indices = get_adsorbate_indices(initial_struc, surface)
            #indices_by_height = sorted(indices, key=lambda i:initial_struc.pos[i,2], reverse=True)
            #ads_indices = indices_by_height[:NUM_ADS_ATOMS]

            #surface_indices1 = indices_by_height[NUM_ADS_ATOMS:NUM_ADS_ATOMS+NUM_SURFACE_ATOMS]
            surface_indices = [i for i in indices if (not torch.eq(initial_struc.pos[i], initial_struc.pos_relaxed[i]).all().item() and i not in ads_indices)]
            #if set(surface_indices1) != set(surface_indices2):
            #    print("Ambiguous surface at", filename)
            #    continue

            if len(ads_indices) == 0:
                print("no adsorbate found at", filename)
                continue

            initial_struc.tags[ads_indices] = 2
            initial_struc.tags[surface_indices] = 1

            ads_energy = sum([ADS_ENERGIES[initial_struc.atomic_numbers[i].int().item()] for i in ads_indices])
            initial_struc.y_init = initial_struc.y - surface_energy - ads_energy # subtract off reference energy, if applicable
            del initial_struc.y
            initial_struc.y_relaxed = relaxed_struc.y - surface_energy - ads_energy # subtract off reference energy, if applicable
            

            
            # Filter data if necessary
            # OCP filters adsorption energies > |10| eV
            
            if ev_bounds is not None and (initial_struc.y_relaxed < ev_bounds[0] or initial_struc.y_relaxed > ev_bounds[1]):
                print("energy out of bounds at", filename)
                continue
            
            initial_struc.sid = idx  # arbitrary unique identifier 
            
            # no neighbor edge case check
            if initial_struc.edge_index.shape[1] == 0:
                print("no neighbors at", filename)
                continue
            
            # Write to LMDB
            txn = db.begin(write=True)
            txn.put(f"{idx}".encode("ascii"), pickle.dumps(initial_struc, protocol=-1))
            txn.commit()
            db.sync()
            idx += 1

    return idx


old_format=True

idx=0
for root_dir in args.run_folders:
    if old_format:
        for facet_dir in get_dirs(root_dir):
            for surface_dir in get_dirs(facet_dir):
                idx = process_surface(surface_dir, idx, ev_bounds=args.ev_bounds)
    else:
        for surface_dir in get_dirs(root_dir):
            idx = process_surface(surface_dir, idx, ev_bounds=args.ev_bounds)

db.close()

