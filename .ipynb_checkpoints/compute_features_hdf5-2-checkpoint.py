import os
import json
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import torch
from torchvision import transforms
import h5py
import timm
from PIL import Image
import pdb
from read_data import * # this might not be neccesary
from resnet import resnet50

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')

    parser.add_argument('--feat_type', default="uni", type=str, required=True, help='Which feature extractor to use, either "resnet" or "uni"')
    parser.add_argument('--ref_file', default="/examples/ref_file.csv", type=str, required=True, help='Path with reference csv file')
    parser.add_argument('--patch_data_path', default="/examples/Patches_hdf5", type=str, required=True, help='Directory where the patch is saved')
    parser.add_argument('--feature_path', type=str, default="/examples/features", help='Output directory to save features')
    parser.add_argument('--max_patch_number', type=int, default=4000, help='Max number of patches to use per slide')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument("--tcga_projects", help="the tcga_projects we want to use", default=None, type=str, nargs='*')
    parser.add_argument('--start', type=int, default=0, help='Start slide index for parallelization')
    parser.add_argument('--end', type=int, default=None, help='End slide index for parallelization')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(10*'-')
    print('Args for this experiment \n')
    print(args)
    print(10*'-')

    random.seed(args.seed)

    path_csv = args.ref_file
    patch_data_path = args.patch_data_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if args.feat_type == 'resnet':
        transforms_val = torch.nn.Sequential(
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])) # these numbers normalise the RGB values of each channel, they represent mean and SD
        
    else:
        transforms_val = transforms.Compose([
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),])

    if args.feat_type == 'resnet': # user picks resnet, this loads it up
        model = resnet50(pretrained=True).to(device)
        ###model= ProjectedResNet(resnet, in_dim=2048, out_dim=1024).to(device)
        model.eval()
    else:
        local_dir = "UNI" # add dir for saved model
        model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, 
                                    init_values=1e-5, num_classes=0, dynamic_img_size=True)
        model.load_state_dict(torch.load(os.path.join(local_dir, 
                                    "pytorch_model.bin"), map_location="cpu"), strict=True)
        model.to(device)
        model.eval()
        x = torch.randn(1, 3, 224, 224).to(device)
        out = model(x)
        print(out.shape)
    
    print('Loading dataset...')

    df = pd.read_csv(path_csv)
    print(df.head(), flush=True)
    df = df.drop_duplicates(["wsi_file_name"]) # there could be duplicated WSIs mapped to different RNA files and we only need features for each WSI
    print(df.head(), flush=True)
    # Filter tcga projects, such as tcga-brca, so that these are the only ones included
    if args.tcga_projects:
        df = df[df['tcga_project'].isin(args.tcga_projects)]

    # indexing based on values for parallelization
    if args.start is not None and args.end is not None: # only one of these if/ elif statements are run at a time
        df = df.iloc[args.start:args.end] # this picks a specific chunk of slides
    elif args.start is not None: # this picks an index and goes to the end
        df = df.iloc[args.start:]
    elif args.end is not None: # this goes from the start to a specific index. only the first if is crucial, the elifs are a bit redundant
        df = df.iloc[:args.end] # for example, if there are 4 GPUs, you could split them up into four groups and each GPU runs each group 

    print(f'Number of slides = {df.shape[0]}') # number of slides processed this run

    for i, row in tqdm(df.iterrows()): # loops through the metadata (like name and project) of each slide. tqdm shows progress bar
        WSI = row['wsi_file_name'] # extracts file name from the reference csv
        WSI_slide = WSI.split('.')[0]
        if args.tcga_projects: ### added this to skip the whole tcga stuff
            project = row['tcga_project'] # similar to 2 lines prior
        WSI = WSI.replace('.svs', '') # in the ref file of prad there is a .svs that should not be there

        if not os.path.exists(os.path.join(patch_data_path, WSI_slide)): # remove those without patch data
            print('Not exist {}'.format(os.path.join(patch_data_path, WSI_slide)))
            continue

        path = os.path.join(patch_data_path, WSI_slide, WSI_slide + '.hdf5') # digs into where the patches from patch_gen_hdf5 are saved
        if args.tcga_projects: ### added this to skip the whole tcga stuff
            path_h5 = os.path.join(args.feature_path, project, WSI) # where they will be next saved. ### only this line remains in the og file out of these 4 lines with ###
        else: ###
            path_h5 = os.path.join(args.feature_path, WSI) ###

        if not os.path.exists(path_h5): # makes the folder if it doesn't exist
            os.makedirs(path_h5)

        if os.path.exists(os.path.join(path_h5, "complete_resnet.txt")): # checks if a certain file is found in the folder, if so feature extraction is skipped
            # the above file name should be "complete_tile.txt"
            print(f'{WSI}: Resnet features already obtained')
            continue

        try:
            with h5py.File(path, 'r') as f_read: # opens the file and gets the patches
                keys = list(f_read.keys())
                if len(keys) > args.max_patch_number:
                    keys = random.sample(keys, args.max_patch_number) # if there are more patches than max_patch_number, a random subset is picked

                features_tiles = []
                for key in tqdm(keys): # reads a patch
                    image = f_read[key][:]
                    if args.feat_type == 'resnet': # extract features
                        image = torch.from_numpy(image).permute(2,0,1)
                        image = transforms_val(image).to(device)
                        with torch.no_grad():
                            features = model.forward_extract(image[None,:]) # forward_extract collects intermediate features
                            features_tiles.append(features[0].detach().cpu().numpy()) # then store in feature_tiles
                    else: # also extract features
                        image = Image.fromarray(image).convert("RGB")
                        image = transforms_val(image).to(device)
                        with torch.no_grad():
                            features = model(image[None,:])
                            features_tiles.append(features[0].detach().cpu().numpy())
                        
            features_tiles = np.asarray(features_tiles)
            n_tiles = len(features_tiles)

            f_write = h5py.File(os.path.join(path_h5, WSI+'.h5'), "w") # save features into hdf5 files, one file per slide
            dset = f_write.create_dataset(f"{args.feat_type}_features", data=features_tiles)
            f_write.close()

            with open(os.path.join(path_h5, "complete_tile.txt"), 'w') as f_sum: # makes a file if successful
                f_sum.write(f"Total n patch = {n_tiles}")

        except Exception as e:
            print(e)
            print(WSI)
            continue

#python sequoia-codes/compute_features_hdf5.py --ref_file sequoia-outputs/ref_file.csv --patch_data_path sequoia-outputs/patches --feature_path sequoia-outputs/uni_features --seed 5 --feat_type uni --tcga_projects COAD --start 0 --end 1

#python sequoia-codes/compute_features_hdf5.py --ref_file sequoia-outputs/ref_file_2.csv --patch_data_path sequoia-outputs/patches --feature_path sequoia-outputs/uni_features --seed 5 --feat_type uni --tcga_projects COAD --start 0 --end 1

#python sequoia-codes/compute_features_hdf5.py --ref_file sequoia-outputs/ref_file_lihc_better_images.csv --patch_data_path sequoia-outputs/patches/ --feature_path sequoia-outputs/uni_features --seed 5 --feat_type uni --tcga_projects LIHC --start 0 --end 1

#python sequoia-codes/compute_features_hdf5.py --ref_file sequoia-outputs/ref_file_coad_multiple_images.csv --patch_data_path sequoia-outputs/patches/ --feature_path sequoia-outputs/uni_features --seed 5 --feat_type uni --tcga_projects COAD --start 0 --end 4
#python sequoia-codes/compute_features_hdf5.py --ref_file sequoia_109/ref_file.csv --patch_data_path sequoia_109/patches/ --feature_path sequoia_109/uni_features --seed 5 --feat_type uni --tcga_projects COAD READ --start 0 --end 109