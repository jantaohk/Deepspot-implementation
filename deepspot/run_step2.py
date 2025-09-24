import pandas as pd
import math
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openslide
import ast
from tqdm import tqdm
from scipy.stats import zscore, pearsonr, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utils import load_slide, get_image_path, convert_cell_table
    
def scaled_tile_sizes(cell, base_tile_size, scale_factors): # this scales patch sizes according to the ratio of number of tumor cells over the number of lymphocytes. This may help with reducing the effect of noise when correlating gene expression with cell distribution. Currently set to 1, so no scaling done. Might be complicated to understand, just keep it as is if in doubt
    counts = cell["cell_label"].value_counts()
    tumor_count = counts.get("tumor", 1)
    lymph_count = counts.get("lymphocyte", 1)
    rel_abundance = tumor_count / lymph_count
    
    return {
        "tumor": [int(base_tile_size)],
        "lymphocyte": [int(base_tile_size* rel_abundance* sf) for sf in scale_factors],
    }
        
def resize_csv_coordinates(csv_file_path, output_path, image_code): # unifies the coordinates system by multiplying by 160. Creates and checks for a flag file to ensure it is run only once
    # need to run this before any of the following functions
    print("Resizing coordinates")        
    df = pd.read_csv(csv_file_path)
    
    if not os.path.exists(os.path.join(output_path, image_code, "complete_resize.txt")):
        df["x"] *= 160
        df["y"] *= 160
        df.rename(columns={'x': 'xcoord', 'y': 'ycoord'}, inplace=True)
        df.to_csv(csv_file_path)
        
        with open(os.path.join(output_path, image_code, "complete_resize.txt"), 'w') as f_sum:
            f_sum.write("csv file coordinates resized")
            
    else:
        print("Already resized, skipping")
        
    return

def st_prediction_heatmap(csv_file_path, output_path, slide_path, image_code, downscale_factor, patch_size, s, base_tile_size, gene_list): # creates ST prediction heatmap
    print("ST predictions")
    save_dir= os.path.join(output_path, image_code)        
    display_spacing= patch_size// downscale_factor
    df= pd.read_csv(csv_file_path).dropna(how= "any")
    df["xcoord"] = df["xcoord"] // downscale_factor
    df["ycoord"] = df["ycoord"] // downscale_factor
    markers = ["CD3D", "CD3E", "CD3G", "CD2", "CD5", "CD7", "CD8A",
               "CD22", "CD37", "CD72", "CD79B", "CD247"]
    df[[f"{m}_z" for m in markers]] = df[markers].apply(zscore)    
    df["CD3_normalised"] = df[["CD3D_z", "CD3E_z", "CD3G_z"]].mean(axis=1)
    df["lymphocytes"] = df[[f"{m}_z" for m in markers]].mean(axis=1)
    slide, w, h, img, *_= load_slide(slide_path, level= 2)
    
    for gene in gene_list: 
        file_name= os.path.join(save_dir, f"{gene}.png")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        sc = ax.scatter(df["xcoord"], df["ycoord"], c=df[gene], cmap="viridis", alpha=1)#, s=s)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.set_label("Predicted Expression")
        ax.set_title(f"{image_code} Predicted {gene} Expression Overlay")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close()

    return

def bulk_rna(csv_file_path, output_path, image_code): # calculates the bulk rna prediction data from ST prediction by taking the average expression of each gene and adding it at the very last row of the csv file
    df= pd.read_csv(csv_file_path)
    bulk_rna = df.iloc[:, 4:].mean()
    new_row = pd.Series([None]*4 + list(bulk_rna), index=df.columns)
    df.loc["bulk_rna"] = new_row 
    file_name= os.path.join(output_path, image_code, f"edited_{image_code}.csv")
    df.to_csv(file_name)    
    return

def rna_correlation_visualisation(info_df, output_path, image_code, summary_df): # correlates the bulk rna prediction data from the previous function with ground truth
    # need to run bulk_rna first to run this
    print("Bulk RNA correlations")
    csv_path= os.path.join(output_path, image_code, f"edited_{image_code}.csv")
    df_csv= pd.read_csv(csv_path)
    csv_avg = df_csv.iloc[-1, 5:]
    csv_avg.index = df_csv.columns[5:]
    rna_df= None
    edited_image_code= "TCGA-"+ image_code
    rna_file_paths= info_df.loc[info_df["patient_barcode"]== edited_image_code, "rna_files"]
    
    if pd.isna(rna_file_paths).all():
        print(f"No RNA files for {image_code}, skipping.")
        
    else:
        for rna_file_str in rna_file_paths:
            rna_file_list = ast.literal_eval(rna_file_str) if isinstance(rna_file_str, str) else rna_file_str
                
            save_dir= os.path.join(output_path, image_code, "bulk_rna_corr")            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                 
            for i, rna_file_path in enumerate(rna_file_list):
                file_name= os.path.join(save_dir, f"{image_code} Spearman correlation {i}.png")
                rna_df = pd.read_csv(rna_file_path, sep="\t", skiprows=1)
                rna_df = rna_df[["gene_name", "tpm_unstranded"]].dropna(subset=["gene_name"])
                rna_df= rna_df[rna_df["tpm_unstranded"]> 0]
                common_genes = rna_df["gene_name"].isin(csv_avg.index)
                rna_df = rna_df.loc[common_genes].copy()
                rna_df["csv_avg"] = rna_df["gene_name"].map(csv_avg)
                rna_df= rna_df.sort_values("gene_name", ascending= True)
                rna_df["tpm_unstranded"] = pd.to_numeric(rna_df["tpm_unstranded"], errors='coerce')
                rna_df["csv_avg"] = pd.to_numeric(rna_df["csv_avg"], errors='coerce')
                rna_df.dropna(subset=["tpm_unstranded", "csv_avg"], inplace=True)
                rna_df["expression_rank"]= rna_df["tpm_unstranded"].rank()
                rna_df["prediction_rank"]= rna_df["csv_avg"].rank()
                rna_df["expression_log"] = np.log1p(rna_df["tpm_unstranded"])
                rna_df["pred_log"] = np.log1p(rna_df["csv_avg"])
                rna_df["rank_diff"] = (rna_df["expression_rank"] - rna_df["prediction_rank"]).abs()
                pearson_corr, pearson_p = pearsonr(rna_df["tpm_unstranded"], rna_df["csv_avg"])
                spearman_corr, spearman_p = spearmanr(rna_df["tpm_unstranded"], rna_df["csv_avg"])  
                plt.figure(figsize=(6,5))
                sns.regplot(x="expression_rank", y="prediction_rank", data=rna_df,
                            scatter_kws={'s':20}, line_kws={'color':'blue'})
                plt.title(f"{image_code} {i} Spearman correlation: {spearman_corr:.3f}")
                plt.xlabel("Expression Rank")
                plt.ylabel("Prediction Rank")
                plt.tight_layout()
                plt.savefig(file_name)
                plt.close()
                logged_pearson_corr, logged_pearson_p = pearsonr(rna_df["expression_log"], rna_df["pred_log"])
                stats = {
                    "pearson_corr": pearson_corr,
                    "pearson_p": pearson_p,
                    "spearman_corr": spearman_corr,
                    "spearman_p": spearman_p,
                    "logged_pearson_corr": logged_pearson_corr,
                    "logged_pearson_p": logged_pearson_p,
                }        
                rna_df = pd.DataFrame([stats])
                rna_df.to_csv(os.path.join(save_dir, f"{image_code} {i}.csv"), index= False)
                summary_df.loc[f"{image_code}-{i}", ["Bulk RNA Pearson", "Bulk RNA Spearman", "Bulk RNA logged Pearson"]]= [pearson_corr, spearman_corr, logged_pearson_corr]
            
    return

def cell_type_analysis(output_path, slide_path, image_code, csv_file_path, downscale_factor, base_tile_size, scale_factors, summary_df, info_df): # extrapolates gene expression to cell type prediction
    print("Cell type analysis") # please see the version in the outdated folder on github. This has been adapted incorrectly and will not run properly
    cell= convert_cell_table(info_df, image_code, slide_path) # this function is wrong
    cell_types= {
        "lymphocyte": ["CD3D", "CD3E", "CD3G"],
        "tumor": ["EPCAM"]
    }
    preds= pd.read_csv(csv_file_path) # you may have to uncomment the below 4 lines. again, see the outdated version
    #preds["xcoord"]= preds["xcoord"]// downscale_factor
    #preds["ycoord"]= preds["ycoord"]// downscale_factor
    #cell["x"]= cell["x"]// 4
    #cell["y"]= cell["y"]// 4
    preds = preds[["xcoord", "ycoord"] + sum(cell_types.values(), [])].dropna() 
    tile_sizes= scaled_tile_sizes(cell, base_tile_size, scale_factors)
    print(tile_sizes)

    for cell_type, markers in cell_types.items():
        sizes_for_type = tile_sizes.get(cell_type, [int(base_tile_size)])
        cell_subset= cell.loc[cell["cell_label"]== cell_type, ["x", "y"]].to_numpy()
            
        for tile_size in sizes_for_type:
            counts= []
                
            for _, row in preds.iterrows():
                x0, y0 = row["xcoord"], row["ycoord"]
                x1, y1 = x0 + tile_size, y0 + tile_size
                
                # query all lymphocytes in the tile
                in_tile= (
                    (cell_subset[:, 0]>= x0)& (cell_subset[:, 0]< x1)&
                    (cell_subset[:, 1]>= y0)& (cell_subset[:, 1]< y1)
                )
                count = np.sum(in_tile)
                counts.append(count)

            preds[f"{cell_type}_count"] = counts
            fig, ax= plt.subplots(1, len(markers), figsize= (5* len(markers), 4))
            
            if len(markers)== 1:
                ax= [ax]
                
            for i, marker in enumerate(markers):
                corr, pval = pearsonr(preds[marker], preds[f"{cell_type}_count"])
                ax[i].scatter(preds[f"{cell_type}_count"], preds[marker], alpha=0.5)
                ax[i].set_xlabel(f"{cell_type.capitalize()} Count per Tile")
                ax[i].set_ylabel(f"Predicted {marker} Expression")
                ax[i].set_title(f"{marker} Pearson r = {corr:.3f}")
                summary_df.loc[image_code, f"{marker}"]= corr
                
            plt.tight_layout() 
            file_name= os.path.join(output_path, image_code, f"{image_code} {cell_type} Pearson correlation.png")
            plt.savefig(file_name)
            plt.close()
            threshold= 0
            slide, W_img, H_img, *_= load_slide(slide_path, level= 0)
            count_map = np.zeros((H_img, W_img), dtype=np.float32)
            
            for _, row in preds.iterrows():
                x = int(row["xcoord"])
                y = int(row["ycoord"])
                v = float(row[f"{cell_type}_count"])
            
                x0, x1 = max(0, x), min(W_img, x + tile_size)
                y0, y1 = max(0, y), min(H_img, y + tile_size)
                
                if x0 < x1 and y0 < y1:
                    count_map[y0:y1, x0:x1] = v

            masked_map = np.where(count_map > threshold, count_map, np.nan)
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(np.array(img), origin="upper")
            hm = ax.imshow(masked_map, origin="upper", alpha=0.5, cmap="viridis")
            ax.set_xlim(0, W_img)
            ax.set_ylim(H_img, 0)
            ax.set_aspect("equal")                
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(hm, cax=cax)
            cbar.set_label(f"{cell_type} Count")                
            ax.set_title(f"{cell_type} Ground Truth Overlay")
            ax.axis("off")
            plt.tight_layout()
            file_name= os.path.join(output_path, image_code, f"{image_code} {cell_type} Ground Truth Overlay.png")
            plt.savefig(file_name)
            plt.close()
            #summary_df.loc[image_code, f"{cell_type}"]= corr
    return
                
def main(output_path, run_steps, gene_list):
    home_dir= os.path.expanduser("~")
    summary_df= pd.DataFrame()
    image_paths, image_codes= get_image_path()
    info_df= pd.read_csv("/home/jantao/109_info.csv") ##### change this path to where your look up table is, created by info_generation.py
    
    for i, slide_path in enumerate(tqdm(image_paths)):
        image_code= image_codes[i]
        slide= img= df= csv_avg= csv_file_path= None
        print(f"Processing {slide_path} image")
        csv_file_path= os.path.join(output_path, image_code, f"{image_code}.csv")
        
        if "load_slide" in run_steps:
            try:
                load_slide(slide_path, level= 2)

            except Exception as e:
                print(f"Error: {e}")
                       
        if "resize_csv" in run_steps:
            try:
                resize_csv_coordinates(csv_file_path, output_path, image_code)
                
            except Exception as e:
                print(f"Error: {e}")
               
        if "st_pred" in run_steps:
            try:
                st_prediction_heatmap(csv_file_path, output_path, slide_path, image_code, 16, 480, 70, 30, gene_list)

            except Exception as e:
                print(f"Error: {e}")
            
        if "bulk_rna" in run_steps:
            try:
                bulk_rna(csv_file_path, output_path, image_code)
                    
            except Exception as e:
                print(f"Error: {e}")
            
        if "rna_corr" in run_steps:
            try:
                rna_correlation_visualisation(info_df, output_path, image_code, summary_df)
                    
            except Exception as e:
                print(f"Error: {e}")
                print("If the above message states file path doesn't exist, please run bulk_rna first.")

        if "cell_type" in run_steps:            
            try:                    
                cell_type_analysis(output_path, slide_path, image_code, csv_file_path, 16, 30, [1], summary_df, info_df)

            except Exception as e:
                print(f"Error: {e}")
                
    #if "rna_corr" and "cell_type" in run_steps:
    if "rna_corr" in run_steps:
        file_name= os.path.join(output_path, "summary.csv")############
        summary_df.to_csv(file_name)
        
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSpot pipeline step 2")
    parser.add_argument("--output_path", required=True, help="Path to save outputs, same as the path from run.py")
    parser.add_argument("--run_steps", nargs="+", default=["resize_csv","st_pred","bulk_rna","rna_corr","cell_type"],
                        help="Pipeline steps to run, can be run individually with some exceptions: resize_csv must be run before downstream functions, and bulk_rna must be run before rna_corr")
    parser.add_argument("--gene_list", nargs="+", default=["EPCAM","CD3D","CD3E","CD3G"],
                        help="List of genes to analyse. See info_highly_variable_genes.csv for the full list")
    args = parser.parse_args()
    output_path= args.output_path
    run_steps= args.run_steps
    gene_list= args.gene_list
    main(output_path= args.output_path,
         run_steps= args.run_steps,
         gene_list= args.gene_list
        )
    
# python DeepSpot/run_step2.py --output_path /home/jantao/deepspot_109 --run_steps resize_csv st_pred bulk_rna rna_corr --gene_list EPCAM CD3D CD3E CD3G
