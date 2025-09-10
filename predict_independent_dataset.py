import os
import json
import argparse
import pickle
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import pandas as pd
import numpy as np # they forgot to import a bunch of stuff

from read_data import SuperTileRNADataset
from utils import filter_no_features, custom_collate_fn
from vit import train, evaluate, predict # only predict is used
from tformer_lin import ViS


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument('--feature_path', type=str, default='', help='Directory where pre-processed WSI features are stored')
    parser.add_argument('--feature_use', type=str, default='cluster_mean_features', help='Which feature to use for training the model')
    parser.add_argument('--folds', type=int, default=5, help='Folds for pre-trained model')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--depth', type=int, default=6, help='Transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--tcga_project', nargs= "+", default=None, help='The tcga_project we want to use') ### think they made a mistake here
    parser.add_argument('--save_dir', type=str, default='', help='Where to save results')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')

    ############################################## variables ##############################################
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    ############################################## saving ##############################################

    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ############################################## data prep ##############################################

    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # filter out WSIs for which we don't have features and filter on TCGA project
    ###df = filter_no_features(df, feature_path = args.feature_path, feature_name = args.feature_use)
    genes = [c[4:] for c in df.columns if "rna_" in c] # extracts gene names
    #genes = list(df.columns[2:]) ### for our case, to extract gene names
    print(f"number of genes must match 20820 {len(genes)}")
    if 'tcga_project' in df.columns and args.tcga_project:
        df = df[df['tcga_project'].isin(args.tcga_project)].reset_index(drop=True) # only keeps the rows for a project, if specified by user
    
    # init test dataloader
    print(args.feature_use)
    test_dataset = SuperTileRNADataset(df, args.feature_path, args.feature_use)
    test_dataloader = DataLoader(test_dataset, 
                                num_workers=0, pin_memory=True, 
                                shuffle=False, batch_size=args.batch_size,
                                collate_fn=custom_collate_fn)
    # wraps into DataLoader for batching and parallelism
    # num_workers: number of background workers
    # pin_memory= True: speeds up data transfer to GPU
    # collate_fn= custom_collate_fn
    feature_dim = test_dataset.feature_dim # accesses dimension of feature vectors of each patch

    res_preds   = [] # lists to store predictions
    res_random  = []
    cancer      = args.tcga_project[0].split('-')[-1].lower() # extracts cancer type from string, only keeping the bits after hyphen and converting to lower case
### made some changes to the line above to accomodate single cancer types
    for fold in range(args.folds):

        # load model from huggingface
        model = ViS.from_pretrained(f"gevaertlab/sequoia-{cancer}-{fold}")
        model.to(device)

        # model prediction on test set
        preds, wsis, projs = predict(model, test_dataloader, run=None)

        # random predictions using a model with randomly initialised weights. This acts as a baseline to assess the proper model's performance
        random_model = ViS(num_outputs=test_dataset.num_genes, 
                            input_dim=feature_dim, 
                            depth=args.depth, nheads=args.num_heads,  
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
        random_model.to(device)
        random_preds, _, _ = predict(random_model, test_dataloader, run=None)

        # save predictions
        res_preds.append(preds)
        res_random.append(random_preds)

    # calculate average across folds
    avg_preds = np.mean(res_preds, axis = 0)
    avg_random = np.mean(res_random, axis = 0)

    df_pred = pd.DataFrame(avg_preds, index = wsis, columns = genes)
    df_random = pd.DataFrame(avg_random, index = wsis, columns = genes)

    test_results = {'pred': df_pred, 'random': df_random}

    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)

#python sequoia-codes/predict_independent_dataset.py --ref_file sequoia-outputs/ref_file.csv --feature_path sequoia-outputs/uni_features --seed 5 --save_dir sequoia-outputs/uni_results --tcga_project COAD --feature_use cluster_features

#python sequoia-codes/predict_independent_dataset.py --ref_file sequoia-outputs/ref_file_2.csv --feature_path sequoia-outputs/uni_features --seed 5 --save_dir sequoia-outputs/uni_results --tcga_project COAD --feature_use cluster_features --exp_name TCGA-3L-AA1B-01Z-00-DX2.17CE3683-F4B1-4978-A281-8F620C4D77B4

#python sequoia-codes/predict_independent_dataset.py --ref_file sequoia-outputs/ref_file_lihc_better_images.csv --feature_path sequoia-outputs/uni_features --seed 5 --save_dir sequoia-outputs/uni_results --tcga_project LIHC --feature_use cluster_features --exp_name TCGA_lihc_frameb

#python sequoia-codes/predict_independent_dataset.py --ref_file sequoia-outputs/ref_file_coad_multiple_images.csv --feature_path sequoia-outputs/uni_features --seed 5 --save_dir sequoia-outputs/uni_results --tcga_project COAD --feature_use cluster_features --exp_name TCGA_coad_multiple_images