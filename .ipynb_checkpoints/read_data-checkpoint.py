import os
import pickle
import random
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import torch
from tqdm import tqdm
import h5py


class SuperTileRNADataset(Dataset):
    def __init__(self, csv_path: str, features_path, feature_use, quick=None): ### they left out feature_use
        self.csv_path = csv_path
        self.quick = quick
        self.features_path = features_path
        self.feature_use= feature_use ### same as above
        if type(csv_path) == str:
            self.data = pd.read_csv(csv_path)
        else:
            self.data = csv_path
        # find the number of genes
        row = self.data.iloc[0]
        self.gene_names= [x for x in row.keys() if "rna_" in x] ### predictions weren't stored properly. this helps somehow
        rna_data = row[self.gene_names].values.astype(np.float32) # check number of genes by checking columns ### this is also changed
        self.num_genes = len(rna_data)

        # find the feature dimension, assume all images in the reference file have the same dimension
        path = os.path.join(self.features_path, row['tcga_project'], 
                            row['wsi_file_name'], row['wsi_file_name']+'.h5')
        f = h5py.File(path, 'r') # gets feature vector size
        features = f[self.feature_use][:] # apparently self.feature_use isn't set here, but in main script, maybe predict_independent_dataset. this might be causing a problem
        self.feature_dim = features.shape[1] 
        f.close()

    def __len__(self):
        return self.data.shape[0] # counts number of samples in the dataset

    def __getitem__(self, idx):
        row = self.data.iloc[idx] # gets row from reference csv
        path = os.path.join(self.features_path, row['tcga_project'],
                            row['wsi_file_name'], row['wsi_file_name']+'.h5') # building the full path for the features of this slide
        rna_data = row[[x for x in row.keys() if 'rna_' in x]].values.astype(np.float32)
        rna_data = torch.tensor(rna_data, dtype=torch.float32) # collects rna expression values and converts into tensor
        try:
            if 'GTEX' not in path:
                path = path.replace('.svs','')
            f = h5py.File(path, 'r')
            features = f['cluster_features'][:] # reads the features for the image
            f.close()
            features = torch.tensor(features, dtype=torch.float32) # converts to tensor
        except Exception as e:
            print(e)
            print(path)
            features = None

        return features, rna_data, row['wsi_file_name'], row['tcga_project']
# features: tensor of number of patches in slide and features
# rna_data: tensor of gene exprssion values
# wsi_file_name: file id for a slide
# tcga_project: project name