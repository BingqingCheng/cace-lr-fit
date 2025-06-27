import ase
import os
import pandas as pd
import numpy as np
import torch
import h5py
import lightning as L

#loader to import dicts:
import cace
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader

class XYZDataset(Dataset):
    def __init__(self,root="data/spice-dipep-dipolar_train.xyz",cutoff=4.0, drop_last=True,
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.cutoff = cutoff
        self.prepare_data()
    
    def prepare_data(self):
        atms = ase.io.read(self.root,index=":")
        data_key = [k for k in atms[0].arrays if k not in ["numbers"]]
        data_key += [k for k in atms[0].info]
        data_key = {k : k for k in data_key}
        dataset = [AtomicData.from_atoms(a,cutoff=self.cutoff,data_key=data_key) for a in atms]
        self.dataset = dataset

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class XYZData(L.LightningDataModule):
    def __init__(self, train_xyz="data/spice-dipep-dipolar_train.xyz",val_xyz="data/spice-dipep-dipolar_val.xyz",test_xyz="data/spice-dipep-dipolar_test.xyz",
                 cutoff=4.0, in_memory=False, drop_last=True, batch_size=1):
        super().__init__()
        self.batch_size = batch_size
        self.train_xyz = train_xyz
        self.val_xyz = val_xyz
        self.test_xyz = test_xyz
        self.drop_last = drop_last
        self.cutoff = cutoff
        try:
            self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            self.num_cpus = os.cpu_count()
        self.prepare_data()
    
    def prepare_data(self):
        self.train = XYZDataset(self.train_xyz,cutoff=self.cutoff)
        self.val = XYZDataset(self.val_xyz,cutoff=self.cutoff)
        self.test = XYZDataset(self.test_xyz,cutoff=self.cutoff)
        
    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
                                  shuffle=True, num_workers = self.num_cpus)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return test_loader

