import pandas as pd
import math
import os
import glob
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openslide
from scipy.stats import zscore, pearsonr, spearmanr
from mpl_toolkits.axes_grid1 import make_axes_locatable

def scaled_tile_sizes(cell, base_tile_size, scale_factors):
    counts = cell["cell_label"].value_counts()
    tumor_count = counts.get("tumor", 1)
    lymph_count = counts.get("lymphocyte", 1)
    rel_abundance = tumor_count / lymph_count
    
    return {
        "tumor": [int(base_tile_size)],
        "lymphocyte": [int(base_tile_size* rel_abundance* sf) for sf in scale_factors],
    }
    
def load_slide(slide_path, level=2):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    #plt.figure()
    #plt.imshow(img)
    #plt.axis("off")
    #plt.show()
    
    return slide, img, w, h
    
def resize_csv_coordinates(csv_file_path, output_path, image_code):
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

def st_prediction_heatmap(csv_file_path, output_path, slide_path, image_code, downscale_factor, patch_size, s, base_tile_size, gene_list):
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
    slide, img, w, h= load_slide(slide_path, level= 2)
    
    for gene in gene_list: 
        file_name= os.path.join(save_dir, f"{gene}.png")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(img)
        sc = ax.scatter(df["xcoord"], df["ycoord"], c=df[gene], cmap="viridis", alpha=1, s=s)
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

def bulk_rna(csv_file_path, output_path, image_code):
    df= pd.read_csv(csv_file_path)
    bulk_rna = df.iloc[:, 4:].mean()
    new_row = pd.Series([None]*4 + list(bulk_rna), index=df.columns)
    df.loc["bulk_rna"] = new_row 
    file_name= os.path.join(output_path, image_code, f"edited_{image_code}.csv")
    df.to_csv(file_name)    
    return

def rna_correlation_visualisation(rna_path, output_path, image_code, summary_df):
    # need to run bulk_rna first to run this
    print("Bulk RNA correlations")
    csv_path= os.path.join(output_path, image_code, f"edited_{image_code}.csv")
    df_csv= pd.read_csv(csv_path)
    csv_avg = df_csv.iloc[-1, 5:]
    csv_avg.index = df_csv.columns[5:]
    rna_df= None
    rna_file_path= os.path.join(rna_path, image_code)
        
    for file in os.listdir(rna_file_path):
        if not file.endswith(".tsv"):
            continue
            
        save_dir= os.path.join(output_path, image_code, "bulk_rna_corr")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
             
        file_name= os.path.join(save_dir, f"{image_code} Spearman correlation.png")
        tsv_path= os.path.join(rna_file_path, file)
        print(f"Processing tsv file: {tsv_path}")
        rna_df = pd.read_csv(tsv_path, sep="\t", skiprows=1)
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
        plt.title(f"{image_code} Spearman correlation: {spearman_corr:.3f}")
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
        rna_df.to_csv(os.path.join(save_dir, f"{image_code}.csv"), index= False)
        summary_df.loc[image_code, ["Bulk RNA Pearson", "Bulk RNA Spearman", "Bulk RNA logged Pearson"]]= [pearson_corr, spearman_corr, logged_pearson_corr]
            
    return

def cell_type_analysis(output_path, cells_path, slide_path, image_code, csv_file_path, downscale_factor, base_tile_size, scale_factors, summary_df):
    print("Cell type analysis")
    matches = glob.glob(os.path.join(cells_path, f"*{image_code}*_cell_table.csv"))
    if len(matches)== 0:
        print("no matches found")
            
    elif len(matches)> 1:
        print("multiple matches found:", matches)
            
    else:   
        cell= matches[0]
        cell= pd.read_csv(cell)
        cell_types= {
            "lymphocyte": ["CD3D", "CD3E", "CD3G"],
            "tumor": ["EPCAM"]
        }
        preds= pd.read_csv(csv_file_path)
        preds["xcoord"]= preds["xcoord"]// downscale_factor
        preds["ycoord"]= preds["ycoord"]// downscale_factor
        cell["x"]= cell["x"]// 4
        cell["y"]= cell["y"]// 4
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
                    
                plt.tight_layout() 
                file_name= os.path.join(output_path, image_code, f"{image_code} {cell_type} Pearson correlation.png")
                plt.savefig(file_name)
                plt.close()
                threshold= 0
                slide, img, w, h= load_slide(slide_path, level= 2)
                W_img, H_img = img.size
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
                summary_df.loc[image_code, f"{cell_type}"]= corr
    return
                
def main(image_path, output_path, rna_path, cells_path, run_steps, gene_list):
    home_dir= os.path.expanduser("~")
    summary_df= pd.DataFrame()
    
    for name in tqdm(os.listdir(image_path)):
        if not name.lower().endswith(".svs"):
            continue
            
        slide= img= df= csv_avg= csv_file_path= None    
        slide_path = os.path.join(image_path, name)
        image_code = "-".join(name.split("-")[1:3])
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
                st_prediction_heatmap(csv_file_path, output_path, slide_path, image_code, 16, 480, 4, 30, gene_list)

            except Exception as e:
                print(f"Error: {e}")
            
        if "bulk_rna" in run_steps:
            try:
                bulk_rna(csv_file_path, output_path, image_code)
                    
            except Exception as e:
                print(f"Error: {e}")
            
        if "rna_corr" in run_steps:
            try:
                rna_correlation_visualisation(rna_path, output_path, image_code, summary_df)
                    
            except Exception as e:
                print(f"Error: {e}")
                print("If the above message states file path doesn't exist, please run bulk_rna first.")

        if "cell_type" in run_steps:            
            try:                    
                cell_type_analysis(output_path, cells_path, slide_path, image_code, csv_file_path, 16, 30, [1], summary_df)

            except Exception as e:
                print(f"Error: {e}")
                
    if "rna_corr" and "cell_type" in run_steps:
        file_name= os.path.join(output_path, "weird images summary.csv")        
        summary_df.to_csv(file_name)
        
if __name__== "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSpot pipeline step 2")
    parser.add_argument("--image_path", required=True, help="Path to folder with .svs images")
    parser.add_argument("--output_path", required=True, help="Path to save outputs, same as the path from run.py")
    parser.add_argument("--rna_path", required=True, help="Path to bulk RNA data, already organised")
    parser.add_argument("--cells_path", required=True, help="Path to cell type data")
    parser.add_argument("--run_steps", nargs="+", default=["load_slide","resize_csv","st_pred","bulk_rna","rna_corr","cell_type"],
                        help="Pipeline steps to run, can be run individually with some exceptions: resize_csv must be run before downstream functions, and bulk_rna must be run before rna_corr")
    parser.add_argument("--genes", nargs="+", default=["EPCAM","CD3D","CD3E","CD3G"],
                        help="List of genes to analyse")
    args = parser.parse_args()
    args.image_path
    output_path= args.output_path
    rna_path= args.rna_path
    cells_path= args.cells_path
    run_steps= args.run_steps
    genes= args.genes
    main(
        image_path=args.image_path,
        output_path=args.output_path,
        rna_path=args.rna_path,
        cells_path=args.cells_path,
        run_steps=args.run_steps,
        gene_list=args.genes
    )
    
# python DeepSpot/run_step2.py --image_path /home/jantao/images/weird_images --output_path /home/jantao/DeepSpot/test_data --rna_path /home/jantao/bulk_rna_data --cells_path "/home/jantao/sequoia-outputs/uni_visualisation/For Jan" --run_steps load_slide resize_csv st_pred bulk_rna rna_corr cell_type --genes EPCAM CD3D CD3E CD3G
