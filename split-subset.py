import argparse
import random
import os
import lmdb
import pickle
from ocpmodels.datasets import SinglePointLmdbDataset, TrajectoryLmdbDataset

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=str)
parser.add_argument("--prefix", type=str)
parser.add_argument("--train_size", type=int)
parser.add_argument("--val_size", type=int)
args = parser.parse_args()

datapath = os.path.join(args.folder, "data.lmdb")
dataset = SinglePointLmdbDataset({"src":datapath})
N = len(dataset)
N_train = args.train_size
N_val, N_test = args.val_size, args.val_size

indices = random.sample(list(range(N)), k=N)
train_data = [dataset[i] for i in indices[:N_train]]
val_data = [dataset[i] for i in indices[N_train:N_train+N_val]]
test_data = [dataset[i] for i in indices[N_train+N_val:N_train+2*N_val]]

def write_db(outdir, examples):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    db = lmdb.open(
        os.path.join(outdir, "data.lmdb"),
        map_size=1099511627776 * 2,
        subdir=False,
        meminit=False,
        map_async=True,
    )
    idx=0
    for ex in examples:
        txn = db.begin(write=True)
        txn.put(f"{idx}".encode("ascii"), pickle.dumps(ex, protocol=-1))
        txn.commit()
        db.sync()
        idx += 1
    db.close()

write_db(os.path.join(args.folder, args.prefix, "train"), train_data)
write_db(os.path.join(args.folder, args.prefix, "val"), val_data)
write_db(os.path.join(args.folder, args.prefix, "test"), test_data)