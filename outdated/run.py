# This code builds upon the DeepSpot code by Nonchev et al. (2025)
# Paper: https://www.medrxiv.org/content/10.1101/2025.02.XX
# Citation:
# @article{nonchev2025deepspot,
#   title={DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H&E Images},
#   author={Nonchev, Kalin and ...},
#   journal={medRxiv},
#   year={2025}
# }
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import dask
dask.config.set({"dataframe.query-planning": True})
from deepspot.utils.utils_image import predict_spot_spatial_transcriptomics_from_image_path, predict_cell_spatial_transcriptomics_from_image_path
from deepspot.utils.utils_image import get_morphology_model_and_preprocess
from deepspot.utils.utils_image import crop_tile
from deepspot.spot.model import DeepSpot
import matplotlib.image as mpimg
import os
from openslide import open_slide
import matplotlib.pyplot as plt
from tqdm import tqdm
import squidpy as sq
import anndata as ad
import pandas as pd
import numpy as np
import multiprocessing as mp
import pandas as pd
import pyvips
import torch
import math
import yaml
import PIL
from PIL import ImageFile, Image
import openslide
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "2" # pick an unoccupied GPU
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSpot pipeline step 1")
    parser.add_argument("--image_path", required=True, help="Path to folder with .svs images")
    parser.add_argument("--output_path", required=True, help="Path to save outputs, same as the path from run_step2.py")
    parser.add_argument("--UNI_path", required=True, help="Path to feature extraction model")
    args = parser.parse_args()
    image_path= args.image_path
    output_path= args.output_path
    UNI_path= args.UNI_path
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    white_cutoff = 200
    model_weights = 'DeepSpot/DeepSpot_pretrained_model_weights/Colon_HEST1K/final_model.pkl'
    model_hparam = 'DeepSpot/DeepSpot_pretrained_model_weights/Colon_HEST1K/top_param_overall.yaml'
    gene_path = 'DeepSpot/DeepSpot_pretrained_model_weights/Colon_HEST1K/info_highly_variable_genes.csv'
    with open(model_hparam, "r") as stream:
        config = yaml.safe_load(stream)
    n_mini_tiles= 9
    spot_diameter= 27
    spot_distance= 30
    downsample_factor= 10
    image_feature_model = config['image_feature_model']
    image_feature_model_path = UNI_path
    genes = pd.read_csv(gene_path)
    selected_genes_bool = genes.isPredicted.values
    genes_to_predict = genes[selected_genes_bool]
    genes_to_predict.sort_values("highly_variable_rank")
    ###genes= genes_to_predict["gene_name"].sort_values(ascending= True).to_list()
    with open(model_hparam, 'r') as yaml_file:
        model_hparam = yaml.safe_load(yaml_file)
    torch.serialization.add_safe_globals([DeepSpot])
    model_expression = torch.load(model_weights, weights_only=False)
    model_expression.to(device)
    model_expression.eval()
    morphology_model, preprocess, feature_dim = get_morphology_model_and_preprocess(model_name=image_feature_model, 
                                                                                    device=device, model_path=image_feature_model_path)
    morphology_model.to(device)
    morphology_model.eval()
    for file in tqdm(os.listdir(image_path), desc= "processing files", unit= "file"):
        if not file.lower().endswith(".svs"):  # skip everything that isn’t .svs
            continue
        svs_path= os.path.join(image_path, file)
        parts= file.split("-")
        sample= f"{parts[1]}-{parts[2]}"
        png_path= os.path.join(output_path, f"{sample}.png")
        slide= openslide.OpenSlide(svs_path)
        level= 2
        dims = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, dims).convert("RGB")
        region.save(png_path, "png")
        image = pyvips.Image.new_from_file(png_path)

        coord = []
        for i, x in enumerate(range(spot_diameter + 1, image.height - spot_diameter - 1, spot_distance)):
            for j, y in enumerate(range(spot_diameter + 1, image.width - spot_diameter - 1, spot_distance)):
                coord.append([i, j, x, y])
        coord = pd.DataFrame(coord, columns=['x_array', 'y_array', 'x_pixel', 'y_pixel'])
        coord.index = coord.index.astype(str)
        is_white = []
        counts = []
        for _, row in tqdm(coord.iterrows(), mininterval= 30):
            x = row.x_pixel - int(spot_diameter // 2)
            y = row.y_pixel - int(spot_diameter // 2)
    
            main_tile = crop_tile(image, x, y, spot_diameter)
            main_tile = main_tile[:,:,:3]
            white = np.mean(main_tile)
            is_white.append(white)

        counts = np.empty((len(is_white), selected_genes_bool.sum())) # empty count matrix 

        coord['is_white'] = is_white
        #coord_filtered = coord[coord['is_white'] <= white_cutoff].copy()###
        #n_spots = len(coord_filtered)###
        #n_genes = len(genes)###
        #counts = np.empty((n_spots, n_genes))###
        adata = ad.AnnData(counts)
        adata.var.index = genes[selected_genes_bool].gene_name.values###
        ###adata.var.index = genes
        #adata.obs = coord_filtered.copy()###
        adata.obs = adata.obs.merge(coord, left_index=True, right_index=True)###
        adata.obs['is_white'] = coord['is_white'].values###
        adata.obs['is_white_bool'] = (coord['is_white'].values > white_cutoff).astype(int)###
        #adata.obs['is_white_bool'] = (coord_filtered['is_white'].values > white_cutoff).astype(int)###
        adata.obs['sampleID'] = sample
        adata.obs['barcode'] = adata.obs.index###
        #adata.obs['barcode'] = adata.obs.index.astype(str)###
        adata = adata[adata.obs.is_white_bool == 0, ]
        img = open_slide(png_path)
        n_level = len(img.level_dimensions) - 1 # 0 based
        
        
        large_w, large_h = img.dimensions
        new_w = math.floor(large_w / downsample_factor)
        new_h = math.floor(large_h / downsample_factor)
        
        whole_slide_image = img.read_region((0, 0), n_level, img.level_dimensions[-1])
        whole_slide_image = whole_slide_image.convert("RGB")
        img_downsample = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
        
        
        adata.obsm['spatial'] = adata.obs[["y_pixel", "x_pixel"]].values
        
        adata.obsm['spatial'] = adata.obsm['spatial'] / downsample_factor
        
        adata.uns['spatial'] = dict()
        adata.uns['spatial']['library_id'] = dict()
        adata.uns['spatial']['library_id']['images'] = dict()
        adata.uns['spatial']['library_id']['images']['hires'] = np.array(img_downsample)
        counts = predict_spot_spatial_transcriptomics_from_image_path(png_path, 
                                                                adata,
                                                                spot_diameter,
                                                                n_mini_tiles,
                                                                preprocess, 
                                                                morphology_model, 
                                                                model_expression, 
                                                                device,
                                                                super_resolution=False,
                                                                neighbor_radius=1)
        counts = model_expression.inverse_transform(counts)
        counts[counts < 0] = 0
        adata_predicted = ad.AnnData(counts, 
                                     var=adata.var,
                                     obs=adata.obs, 
                                     uns=adata.uns, 
                                     obsm=adata.obsm).copy()
        genes_present= adata_predicted.var_names###
        spatial_coords = pd.DataFrame(
            adata_predicted.obsm['spatial'], 
            columns=['x', 'y'],
            index=adata_predicted.obs_names
        )
        

        expr_data = adata_predicted[:, genes_present].X
        if hasattr(expr_data, "toarray"):
            expr_data = expr_data.toarray()
        
        expr_df = pd.DataFrame(expr_data, columns= genes_present, index=adata_predicted.obs_names)
        
        combined_df = pd.concat([spatial_coords, expr_df], axis=1)
        save_path= os.path.join(output_path, sample)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        combined_df.to_csv(f"{save_path}/{sample}.csv")
        
# python DeepSpot/run.py --image_path /home/jantao/images/coad_images --output_path /home/jantao/DeepSpot/test_data --UNI_path /home/jantao/UNI/pytorch_model.bin
