import os
import json
import argparse
import pdb
from tqdm import tqdm

import numpy as np
import h5py
from sklearn.cluster import KMeans
import torch.nn as nn
import torch
from read_data import * # might also not be needed
from utils import exists


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--feat_type', default="uni", type=str, required=True, help='Which feature extractor to use, either "resnet" or "uni"')
    parser.add_argument('--ref_file', default="/examples/ref_file.csv", type=str, required=True, help='Path with reference csv file')
    parser.add_argument('--patch_data_path', default="/examples/Patches_hdf5", type=str, required=True, help='Directory where the patch is saved')
    parser.add_argument('--feature_path', type=str, default="/examples/features", help='Output directory to save features')
    parser.add_argument('--num_clusters', type=int, default=100,
                        help='Number of clusters for the kmeans')
    parser.add_argument("--tcga_projects", help="the tcga_projects we want to use",
                        default=None, type=str, nargs='*')
    parser.add_argument('--start', type=int, default=0,
                        help='Start slide index for parallelization')
    parser.add_argument('--end', type=int, default=None,
                        help='End slide index for parallelization')
    parser.add_argument("--gtex", help="using gtex data", # i think this gives you the option to specify tissue
                    action="store_true")
    parser.add_argument('--gtex_tissue', type=str, default=None, # this names the tissue
                        help='GTex tissue being used')
    parser.add_argument('--seed', type=int, default=99,
        help='Seed for random generation')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(10*'-')
    print('Args for this experiment \n')
    print(args)
    print(10*'-')

    path_csv = args.ref_file

    df = pd.read_csv(path_csv)
    df = df.drop_duplicates(['wsi_file_name'])

    # Filter tcga projects out of other projects
    if args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)]

    print(f'Total number of slides = {df.shape[0]}')

    # indexing based on values for parallelization (same as in a previous script)
    if exists(args.start) and exists(args.end):
        df = df.iloc[args.start:args.end]
    elif exists(args.start):
        df = df.iloc[args.start:]
    elif args.end:
        df = df.iloc[:args.end]

    print(f'New number of slides = {df.shape[0]}')

    for i, row in tqdm(df.iterrows()): # looping through each row
        WSI = row['wsi_file_name']
        if args.gtex:
            project = args.gtex_tissue
            path = os.path.join(args.feature_path, project, WSI) ### this was moved here, too complicated to explain how, refer to github
        elif args.tcga_projects: ### added to exclude tcga stuff
            project = df.iloc[i]['tcga_project']
            WSI = WSI.replace('.svs', '')
            path = os.path.join(args.feature_path, project, WSI) ### this was moved here, too complicated to explain how, refer to github
        else: ### the entire else statement is added
            project= "kmeans features"
            path = os.path.join(args.feature_path, WSI)

        
        try:
            f = h5py.File(os.path.join(path,WSI+'.h5'), "r+") # opens the hdf5 file containing the features
        except:
            print(f'Cannot open file {path}')
            continue
        try:
            if args.feat_type== "resnet":
                features = f['resnet_features'] # seems like uni might be discontinued
            else:
                features= f["uni_features"]
        except:
            print(f'No resnet features for {path}')
            f.close()
            continue

        if features.shape[0] < args.num_clusters: # skip if there are fewer patches than clusters
            print(f'{WSI} less number of patches than clusters')
            f.close()
            continue

        if 'cluster_features' in f.keys(): # skip if already done
            print(f'{WSI}: Cluster feature already available')
            f.close()
            continue

        kmeans = KMeans(n_clusters=args.num_clusters, random_state=0).fit(features)
        clusters = kmeans.labels_

        mean_features = []
        for pos in tqdm(range(args.num_clusters)): # for each cluster get the patch features and compute the mean feature vector
            indexes = np.where(clusters == pos)
            features_aux = features[indexes]
            mean_features.append(np.mean(features_aux, axis=0)) # assign mean vectors to mean_features

        mean_features = np.asarray(mean_features)
        try: # save
            dset = f.create_dataset("cluster_features", data=mean_features)
            f.close()
        except Exception as e:
            print(f"{WSI}: Error in creating cluster_feauture")
            print(e)
            f.close()
    print('Done!')

#python sequoia-codes/kmean_features.py --ref_file sequoia-outputs/ref_file.csv --patch_data_path sequoia-outputs/patches --feature_path sequoia-outputs/uni_features --seed 5 --tcga_projects COAD --feat_type uni --start 0 --end 1

#python sequoia-codes/kmean_features.py --ref_file sequoia-outputs/ref_file_2.csv --patch_data_path sequoia-outputs/patches --feature_path sequoia-outputs/uni_features --seed 5 --tcga_projects COAD --feat_type uni --start 0 --end 1

#python sequoia-codes/kmean_features.py --ref_file sequoia-outputs/ref_file_lihc_better_images.csv --patch_data_path sequoia-outputs/patches/TCGA_lihc_frameb --feature_path sequoia-outputs/uni_features --seed 5 --tcga_projects LIHC --feat_type uni --start 0 --end 1

#python sequoia-codes/kmean_features.py --ref_file sequoia-outputs/ref_file_coad_multiple_images.csv --patch_data_path sequoia-outputs/patches/coad_multiple_images --feature_path sequoia-outputs/uni_features --seed 5 --tcga_projects COAD --feat_type uni --start 0 --end 4