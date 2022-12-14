import argparse
import random
import os
import lmdb
import pickle

import bisect
import random
from pathlib import Path

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric

def pyg2_data_transform(data: Data):
    """
    if we're on the new pyg (2.0 or later) and if the Data stored is in older format
    we need to convert the data to the new format
    """
    if torch_geometric.__version__ >= "2.0" and "_store" not in data.__dict__:
        return Data(
            **{k: v for k, v in data.__dict__.items() if v is not None}
        )

    return data

class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, config, transform=None):
        super(LmdbDataset, self).__init__()
        self.config = config

        self.path = Path(self.config["src"])
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            self._keys = [
                f"{j}".encode("ascii")
                for j in range(self.env.stat()["entries"])
            ]
            self.num_samples = len(self._keys)

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled))

        if self.transform is not None:
            data_object = self.transform(data_object)

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--out_folder", type=str, default="")
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    args = parser.parse_args()

    datapath = os.path.join(args.folder, "data.lmdb")
    dataset = LmdbDataset({"src":datapath})
    N = len(dataset)
    N_val, N_test = int(args.val_frac * N), int(args.test_frac * N)
    N_train = N - N_val - N_test

    indices = random.sample(list(range(N)), k=N)
    train_data = [dataset[i] for i in indices[:N_train]]
    val_data = [dataset[i] for i in indices[N_train:N_train+N_val]]
    test_data = [dataset[i] for i in indices[N_train+N_val:]]

    outfolder = args.out_folder if len(args.out_folder) > 0 else args.folder
    if len(train_data) > 0:
        write_db(os.path.join(outfolder, "train"), train_data)
    if len(val_data) > 0:
        write_db(os.path.join(outfolder, "val"), val_data)
    if len(test_data) > 0:
        write_db(os.path.join(outfolder, "test"), test_data)



'''
def get_data_size(path):
    dataset = splitdata.LmdbDataset({"src":os.path.join(path, "data.lmdb")})
    n = len(dataset)
    dataset.close_db()
    return n

for dir in os.listdir('./'):
    try:
        for split in ["train", "val", "test"]:
            dirpath = os.path.join(dir, split)
            n = get_data_size(dirpath)
            print(dirpath + ": " + str(n))
    except:
        continue
'''