
from pathlib import Path
from typing import Sequence, Union, Dict

import os
import pickle
from functools import lru_cache

import lmdb

import numpy as np
import torch
from torch import Tensor


class LMDBDataset:
    def __init__(self, db_path):
        super().__init__()
        assert Path(db_path).exists(), f"{db_path}: No such file or directory"
        self.env = lmdb.Environment(
            db_path,
            map_size=(1024 ** 3) * 256,
            subdir=False,
            readonly=True,
            readahead=True,
            meminit=False,
            lock=True
        )
        self.len: int = self.env.stat()["entries"]

    def __len__(self):
        return self.len

    @lru_cache(maxsize=16)
    def __getitem__(self, idx: int) -> Dict[str, Union[Tensor, float]]:
        if idx < 0 or idx >= self.len:
            raise IndexError
        data = pickle.loads(self.env.begin().get(f"{idx}".encode()))
        return data


def split_data_by_element(dataset, element_list):
    remainder = []
    sets_by_element = {e:[] for e in element_list}

    for ex in dataset:
        ex_elements = ex["atomic_numbers"].long().unique()
        keep=True
        for element in element_list:
            if element in ex_elements:
                sets_by_element[element].append(ex)
                keep=False
        if keep:
            remainder.append(ex)
    
    return remainder, sets_by_element

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
    train_path = "/scratch/hdd001/home/kaselby/ocp/data/is2re/all/train/data.lmdb"
    db = LMDBDataset(train_path)

    element_list = [17,37,81,75,55,5]
    remainder, sets_by_element = split_data_by_element(db, element_list)

    outdir="/scratch/hdd001/home/kaselby/ocp/element_data/1"
    write_db(os.path.join(outdir, "train"), remainder)
    for element in element_list:
        write_db(os.path.join(outdir, str(element)), sets_by_element[element])