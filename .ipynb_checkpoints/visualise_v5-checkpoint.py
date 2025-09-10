import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,4"
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from einops import rearrange
import pickle
from scipy.ndimage import binary_dilation
import openslide
from PIL import Image
import timm
import torch
import time
from torchvision import transforms
from he2rna import HE2RNA
from vit import ViT
from resnet import resnet50
from tformer_lin import ViS
from safetensors.torch import load_file
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
Image.MAX_IMAGE_PIXELS = None

BACKGROUND_THRESHOLD = .5


def read_pickle(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


def sliding_window_method(df, patch_size_resized, 
                            feat_model, model, inds_gene_of_interest, stride, 
                            feat_model_type, feat_dim, model_type, slide, device='cpu'):

    max_x = max(df['xcoord_tf'])
    max_y = max(df['ycoord_tf'])
    #debug_counter= 0
    #max_debug_saves= 5
    preds = {} # {key:value} where key is a gene index and value is a new dict that contains the predictions per tile for that gene
    for ind_gene in inds_gene_of_interest:
        preds[ind_gene] = {}
    patch_cache= {}        
    for x in tqdm(range(0, max_x, stride)):
        if x > 0:
            x_to_remove = min(df['xcoord']) + (x - 1) * patch_size_resized
            keys_to_remove = [key for key in patch_cache if key[0] == x_to_remove]
            for key in keys_to_remove:
                patch_cache.pop(key)
                
        for y in range(0, max_y, stride):
            
            window = df[((df['xcoord_tf']>=x) & (df['xcoord_tf']<(x+10))) &
                        ((df['ycoord_tf']>=y) & (df['ycoord_tf']<(y+10)))]

            if window.shape[0] > ((10*10)/2):
                # get the patches
                #for ind in window.index:
                 #   col = df.iloc[ind]['xcoord']
                  #  row = df.iloc[ind]['ycoord']
                cols= window["xcoord"].values
                rows= window["ycoord"].values
                window_keys = [(col, row) for col, row in zip(cols, rows)]
                uncached_keys= []
                uncached_patches= []
                for col, row in zip(cols, rows):
                    key= (col, row)

                    if key not in patch_cache:
                        if hasattr(slide, "read_region"):                            
                            patch = slide.read_region((col, row), 0, (patch_size_resized, patch_size_resized)).convert('RGB')
                        else:
                            patch= slide.crop((col, row, col + patch_size_resized, row + patch_size_resized)).convert('RGB')
                        uncached_keys.append(key)
                        uncached_patches.append(patch)
                if uncached_patches:
                    patches_tf= torch.stack([transforms_(p) for p in uncached_patches])
                    for k, tf_patch in zip(uncached_keys, patches_tf):
                        patch_cache[k]= tf_patch

                batch = torch.stack([patch_cache[k] for k in window_keys]).to(device)
                with torch.no_grad():
                    if feat_model_type == 'resnet':
                        features_all = feat_model.forward_extract(batch)
                    else:
                        features_all = feat_model(batch)

                if features_all.shape[0] < 100:
                    pad_len= 100- features_all.shape[0]
                    padding= torch.zeros(pad_len, feat_dim, device= features_all.device)
                    features_all = torch.cat([features_all, padding])

                # get predictions
                with torch.no_grad():
                    if model_type == 'he2rna':
                        features_all = torch.unsqueeze(features_all, dim=0)
                        features_all = rearrange(features_all, 'b c f -> b f c')
                    model_predictions = model(features_all)
                    
                predictions = model_predictions.detach().cpu().numpy()[0]

                # add predictions to dict (same for all tiles in window)
                for ind_gene in inds_gene_of_interest:
                    for _, key in enumerate(window.index):
                        if stride == 10:
                            preds[ind_gene][key] = predictions[ind_gene] 
                        else:
                            if key not in preds[ind_gene].keys():
                                preds[ind_gene][key] = [predictions[ind_gene]]
                            else:
                                preds[ind_gene][key].append(predictions[ind_gene])

    if stride < 10:
        for ind_gene in inds_gene_of_interest:
            for key in preds[ind_gene].keys():
                preds[ind_gene][key] = np.mean(preds[ind_gene][key])
    return preds

if __name__=='__main__':

    print('Start running visualize script')

    ############################## get args
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--study', type=str, help='cancer study abbreviation, lowercase')
    parser.add_argument('--project', type=str, help='name of project (spatial_GBM_pred, TCGA-GBM, PESO, Breast-ST)')
    parser.add_argument('--gene_names', type=str, help='name of genes to visualize, separated by commas. if you want all the predicted genes, pass "all" ')
    parser.add_argument('--wsi_file_name', type=str, help='wsi filename')
    parser.add_argument('--save_folder', type=str, help='destination folder')
    parser.add_argument('--model_type', type=str, help='model to use:  "he2rna", "vit" or "vis"')
    parser.add_argument('--feat_type', type=str, help='"resnet" or "uni"')
    parser.add_argument('--folds', type=list, help='folds to use in prediction split by comma', default='0,1,2,3,4')
    parser.add_argument("--checkpoint", type= str)
    parser.add_argument("--stride", type= int, default= 1)
    parser.add_argument("--patch_size", type= int, default= 256)
    parser.add_argument("--mask", type= str)
    parser.add_argument("--save_name", type= str)
    args = parser.parse_args()

    ############################## general 
    study = args.study
    assert args.feat_type in ['resnet', 'uni']
    assert args.model_type in ['vit', 'vis', 'he2rna']

    ############################## get model
    ###checkpoint = f'{args.model_type}_{args.feat_type}/{study}/'
    ###obj = read_pickle(checkpoint + 'test_results.pkl')[0]
    ###gene_ids = obj['genes']
    checkpoint= args.checkpoint
    obj = read_pickle(os.path.join(checkpoint, "test_results.pkl"))[0]
    gene_ids = list(obj['pred'].columns) ### try removing list() if problems arise


    ############################## prepare data
    stride = args.stride ### originally 1
    patch_size = args.patch_size # at 20x (0.5um pp) ### or 256? (og)
    wsi_file_name = args.wsi_file_name
    project = args.project 
    ###save_path = f'./visualizations/{project}/{args.save_folder}/{args.wsi_file_name}/'
    save_path= "sequoia-outputs/uni_visualisation"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gene_names != 'all':
        if '.npy' in args.gene_names:
            gene_names = np.load(args.gene_names,allow_pickle=True)
        else:
            gene_names = args.gene_names.split(",")
    else:
        gene_names = gene_ids
        
    # prepare and load WSI
    if 'TCGA' in wsi_file_name:
        ###slide_path = './TCGA/'+project+'/'
        ###mask_path = './TCGA/'+project+'_Masks/'
        mask= np.load(args.mask)
        print("np format, (y, x)")
        print(f"mask dimensions: {mask.shape}")
        manual_resize = None # nr of um/px can be read from slide properties
    else:
        print('please provide correct file name format (containing "TCGA") or correct project id ("spatial_GBM_pred" or "Breast-ST")')
        exit()
    
    # load wsi and calculate patch size in original image (coordinates are at level 0 for openslide) and in mask
    ###slide = openslide.OpenSlide(slide_path + wsi_file_name)
    slide_path= args.wsi_file_name
    try:        
        slide= openslide.OpenSlide(slide_path)
        downsample_factor = int(slide.dimensions[0]/mask.shape[0]) # mask downsample factor
        slide_dim0, slide_dim1 = slide.dimensions[0], slide.dimensions[1]

        if manual_resize == None:
            resize_factor = float(slide.properties.get('aperio.AppMag',20)) / 20.0
        else:
            resize_factor = manual_resize
    except Exception as e:
        slide= Image.open(slide_path)
        print(f"slide.size: {slide.size}")
        print(f"mask.shape: {mask.shape}")
        downsample_factor= int(slide.size[0]/mask.shape[0])
        slide_dim0, slide_dim1= slide.size[0], slide.size[1]
        
        if manual_resize== None:
            resize_factor= 1
        else:
            resize_factor= manual_resize
            
    patch_size_resized = int(resize_factor * patch_size)
    print(f"patch_size_resized is {patch_size_resized}")
    patch_size_in_mask = int(patch_size_resized/downsample_factor)

    # get valid coordinates (that have tissue)
    valid_idx = []
    mask = (np.transpose(mask, axes=[1,0]))* 1
    for col in range(0, slide_dim0-patch_size_resized, patch_size_resized): #slide.dimensions is (width, height)
        for row in range(0, slide_dim1-patch_size_resized, patch_size_resized):
            row_downs = int(row/downsample_factor)
            col_downs = int(col/downsample_factor)

            patch_in_mask = mask[row_downs:row_downs+patch_size_in_mask,col_downs:col_downs+patch_size_in_mask]
            patch_in_mask = binary_dilation(patch_in_mask, iterations=3)

            if patch_in_mask.sum() >= (BACKGROUND_THRESHOLD * patch_in_mask.size):
                # keep patch
                valid_idx.append((col, row))
    # dataframe which contains coordinates of valid patches
    df = pd.DataFrame(valid_idx, columns=['xcoord', 'ycoord'])
    # rescale coordinates to (0,0) and step size 1    
    df['xcoord_tf'] = ((df['xcoord']-min(df['xcoord']))/patch_size_resized).astype(int)
    df['ycoord_tf'] = ((df['ycoord']-min(df['ycoord']))/patch_size_resized).astype(int)

    print('Got dataframe of valid tiles')

    ############################## feature extractor model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device= torch.device("cuda:0")
    device_str= "cuda:0"
    if args.feat_type == 'resnet':
        transforms_ = transforms.Compose([
            transforms.Resize((256,265)), ### is this correct?
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        feat_model = resnet50(pretrained=True).to(device)
        feat_model.eval()
    else:
        feat_model = timm.create_model("vit_large_patch16_224", img_size=224, 
                                        patch_size=16, init_values=1e-5, 
                                        num_classes=0, dynamic_img_size=True)
        local_dir = "UNI"
        feat_model.load_state_dict(torch.load(os.path.join(local_dir, 
                                    "pytorch_model.bin"), map_location=device), strict=True)
        transforms_ = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        feat_model = feat_model.to(device)
        feat_model.eval()

    ############################## get preds
    res_df = df.copy(deep=True)
    ###folds = [int(i) for i in args.folds.split(',')]
    folds= args.folds
    for fold in folds:
        if fold== "0":
            fold_ckpt= "sequoia-codes/lihc_model.safetensors" ###########################
            #fold_ckpt= "sequoia-codes/model.safetensors"
        elif fold== "1":
            fold_ckpt= "sequoia-codes/lihc_model (1).safetensors"
        elif fold== "2":
            fold_ckpt= "sequoia-codes/lihc_model (2).safetensors"
        elif fold== "3":
            fold_ckpt= "sequoia-codes/lihc_model (3).safetensors"
        elif fold== "4":
            fold_ckpt= "sequoia-codes/lihc_model (4).safetensors"
        ###fold_ckpt = checkpoint + 'model_best_' + str(fold) + '.pt'
        ###if (fold == 0) and ((args.model_type == 'vit') or (args.model_type == 'vis')):
           ### fold_ckpt = fold_ckpt.replace('_0','')

        ###input_dim = 2048 if args.feat_type == 'resnet' else 1024 ### could be problematic
        input_dim= 1024
        state_dict = load_file(fold_ckpt, device='cpu') ### for running on multiple gpus (2)
        new_state_dict = {} ###
        for k, v in state_dict.items(): ###
            new_key = 'module.' + k ###
            new_state_dict[new_key] = v ###
            
        if args.model_type == 'vit':
            model = ViT(num_outputs=len(gene_ids), 
                            ###same here dim=input_dim, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
                            dim=input_dim, depth=6, heads=16, mlp_dim=1024, dim_head = 64)
            model= torch.nn.DataParallel(model, device_ids= [0, 1]) ###
            model.load_state_dict(new_state_dict) ###
        elif args.model_type == 'he2rna':
            model = HE2RNA(input_dim=input_dim, layers=[256,256],
                                ks=[1,2,5,10,20,50,100],
                                output_dim=len(gene_ids), device=device)
            fold_ckpt = fold_ckpt.replace('best_','')
            model.load_state_dict(torch.load(fold_ckpt, map_location=torch.device(device)).state_dict())
        elif args.model_type == 'vis':
            model = ViS(num_outputs=len(gene_ids), 
                        input_dim=input_dim, 
                        depth=6, nheads=16,  
                        dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            model= torch.nn.DataParallel(model, device_ids= [0, 1]) ###
            ###model.load_state_dict(torch.load(fold_ckpt, map_location=torch.device(device)).state_dict())
            model.load_state_dict(new_state_dict) ###

        model = model.to(device)
        model.eval()

        # get indices of requested genes
        inds_gene_of_interest = []
        for gene_name in gene_names:
            try:
                inds_gene_of_interest.append(gene_ids.index(gene_name))
            except:
                print('gene not in predicted values '+gene_name)
        
        # get visualization
        preds = sliding_window_method(df=df, patch_size_resized=patch_size_resized, 
                                        feat_model=feat_model, model=model, 
                                        inds_gene_of_interest=inds_gene_of_interest, stride=stride,
                                        feat_model_type=args.feat_type, feat_dim=input_dim, model_type=args.model_type, slide= slide, device=device)
        new_cols= { ### added the following and deleted below for efficiency
            gene_ids[ind_gene] + f'_{fold}': res_df.index.map(preds[ind_gene]) ###
            for ind_gene in inds_gene_of_interest ###
        } ###
        res_df = pd.concat([res_df, pd.DataFrame(new_cols, index=res_df.index)], axis=1)
        ###for ind_gene in inds_gene_of_interest:
            ###res_df[gene_ids[ind_gene] + '_' + str(fold)] = res_df.index.map(preds[ind_gene])

    for ind_gene in inds_gene_of_interest:
        res_df[gene_ids[ind_gene]] = res_df[[gene_ids[ind_gene] + '_' + str(i) for i in folds]].mean(axis=1)

    save_name = os.path.join(save_path, args.save_name) ####################
    res_df.to_csv(save_name)

    print('Done')
#python sequoia-codes/visualize_v5.py --study coad --project spatial_coad_pred --wsi_file_name images/coad_images/TCGA-4N-A93T-01Z-00-DX2.875E7F95-A6D4-4BEB-A331-F9D8080898C2.svs --gene_names ALB,CD68,EPCAM,CD3D,CD3E,CD3G,KRT19 --save_folder sequoia-outputs/uni_visualisation --model_type vis --feat_type uni --folds 0 --checkpoint sequoia-outputs/uni_results/TCGA_coad_multiple_images --patch_size 512 --stride 1 --mask sequoia-outputs/masks/TCGA_coad_multiple_images/TCGA-4N-A93T-01Z-00-DX2/mask.npy --save_name "4N-A93T 512.csv"
#python sequoia-codes/visualise_v5.py --study coad --project spatial_coad_pred --wsi_file_name images/coad_images/TCGA-4T-AA8H-01Z-00-DX1.A46C759C-74A2-4724-B6B5-DECA0D16E029.svs --gene_names ALB,CD68,EPCAM,CD3D,CD3E,CD3G,KRT19 --save_folder sequoia-outputs/uni_visualisation --model_type vis --feat_type uni --folds 0 --checkpoint sequoia-outputs/uni_results/TCGA_coad_multiple_images --patch_size 512 --stride 1 --mask sequoia-outputs/masks/TCGA_coad_multiple_images/TCGA-4T-AA8H-01Z-00-DX1/mask.npy --save_name "4T-AA8H 512.csv"