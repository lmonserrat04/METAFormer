import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SingleAtlas(Dataset):
    def __init__(self, df, augment=0.0):
        self.augment = augment
        print("Cargando SingleAtlas en RAM...")
        self.x      = [torch.tensor(np.loadtxt(row.aal)).float()      for _, row in tqdm(df.iterrows(), total=len(df))]
        self.labels = [row.LABELS                                       for _, row in df.iterrows()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.x[idx].clone()

        if np.random.rand() > self.augment:
            x += torch.randn_like(x) * 0.01

        x = (x - x.mean()) / x.std()

        label = torch.tensor([self.labels[idx]])
        label = torch.eye(2)[label].float().squeeze()

        return x, label


class MultiAtlas(Dataset):
    def __init__(self, df, augment=0.0):
        self.augment = augment
        print("Cargando MultiAtlas en RAM...")
        self.aal    = [torch.tensor(np.loadtxt(row.aal)).float()          for _, row in tqdm(df.iterrows(), total=len(df))]
        self.cc200  = [torch.tensor(np.loadtxt(row.cc200)).float()        for _, row in tqdm(df.iterrows(), total=len(df))]
        self.do160  = [torch.tensor(np.loadtxt(row.dosenbach160)).float() for _, row in tqdm(df.iterrows(), total=len(df))]
        self.labels = [row.LABELS                                          for _, row in df.iterrows()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        aal   = self.aal[idx].clone()
        cc200 = self.cc200[idx].clone()
        do160 = self.do160[idx].clone()

        if np.random.rand() > self.augment:
            aal   += torch.randn_like(aal)   * 0.01
            cc200 += torch.randn_like(cc200) * 0.01
            do160 += torch.randn_like(do160) * 0.01

        aal   = (aal   - aal.mean())   / aal.std()
        cc200 = (cc200 - cc200.mean()) / cc200.std()
        do160 = (do160 - do160.mean()) / do160.std()

        label = torch.tensor([self.labels[idx]])
        label = torch.eye(2)[label].float().squeeze()

        return (aal, cc200, do160), label


class ImputationDataset(Dataset):
    def __init__(self, df, mask_ratio=0.15):
        self.mask_ratio = mask_ratio
        print("Cargando ImputationDataset en RAM...")
        self.aal    = [torch.tensor(np.loadtxt(row.aal)).float()          for _, row in tqdm(df.iterrows(), total=len(df))]
        self.cc200  = [torch.tensor(np.loadtxt(row.cc200)).float()        for _, row in tqdm(df.iterrows(), total=len(df))]
        self.do160  = [torch.tensor(np.loadtxt(row.dosenbach160)).float() for _, row in tqdm(df.iterrows(), total=len(df))]

    def __len__(self):
        return len(self.aal)

    def __getitem__(self, idx):
        aal   = self.aal[idx].clone()
        cc200 = self.cc200[idx].clone()
        do160 = self.do160[idx].clone()

        aal   = (aal   - aal.mean())   / aal.std()
        cc200 = (cc200 - cc200.mean()) / cc200.std()
        do160 = (do160 - do160.mean()) / do160.std()

        aal_mask = torch.tensor(np.random.choice(
            [0, 1], size=aal.shape,   p=[1-self.mask_ratio, self.mask_ratio]))
        cc200_mask = torch.tensor(np.random.choice(
            [0, 1], size=cc200.shape, p=[1-self.mask_ratio, self.mask_ratio]))
        do160_mask = torch.tensor(np.random.choice(
            [0, 1], size=do160.shape, p=[1-self.mask_ratio, self.mask_ratio]))

        aal_masked   = aal   * ~aal_mask.bool()
        cc200_masked = cc200 * ~cc200_mask.bool()
        do160_masked = do160 * ~do160_mask.bool()

        return (aal, cc200, do160), (aal_masked, cc200_masked, do160_masked), (aal_mask, cc200_mask, do160_mask)