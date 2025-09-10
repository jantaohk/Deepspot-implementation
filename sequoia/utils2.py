import subprocess
import os
import pandas as pd
import pickle
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openslide
from scipy.stats import zscore, pearsonr, spearmanr
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_slide(slide_path, level=2):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    mpp_x = float(slide.properties.get("openslide.mpp-x", "nan"))
    mpp_y = float(slide.properties.get("openslide.mpp-y", "nan"))
    return slide, img, w, h, mpp_x, mpp_y
    
def ref_file_gen(output_path):
    ref_file_path= os.path.join(output_path, "ref_file.csv")
    
    if not os.path.exists(os.path.join(output_path, "complete_resize.txt")):
        print("Generating ref_file.csv")
        info_path= "/home/jantao/109_info.csv"
        df= pd.read_csv(info_path)
        ref_df= df[["data_id", "patient_barcode", "cancer_type"]]
        ref_df.rename(columns= {"data_id": "wsi_file_name", "patient_barcode": "patient_id", "cancer_type": "tcga_project"}, inplace= True)
        dataframe= pd.read_csv("/home/jantao/ref_file.csv") # downloaded from sequoia repo
        dataframe = dataframe.drop(columns=['wsi_file_name', 'patient_id', "tcga_project"])
        dataframe = dataframe.iloc[0:0]
        ref_df= pd.concat([ref_df, dataframe], axis=1)
        ref_file_path= os.path.join(output_path, "ref_file.csv")
        ref_df.to_csv(ref_file_path)
        
        with open(os.path.join(output_path, "complete_resize.txt"), 'w') as f_sum:
            f_sum.write("ref_file created")

    else:
        print("ref_file already created, skipping")
        
    return ref_file_path
    
def patch_gen(info_csv, patches, masks):
    print("Generating patches")
    cmd= [
        "python", "sequoia-codes/patch_gen_hdf5-2.py",
        "--lookup_table", info_csv,
        "--patch_path", patches,
        "--mask_path", masks,
        "--start", "0",
        "--end", "109",
        "--max_patches_per_slide", "4000",
        "--parallel", "1",
        "--patch_size", "256"
    ]
    subprocess.run(cmd, check= True)
    return

def compute_feat(ref_file_path, patches, uni_features):
    print("Computing features")
    cmd= [
        "python", "sequoia-codes/compute_features_hdf5-2.py",
        "--ref_file", ref_file_path,
        "--patch_data_path", patches,
        "--feature_path", uni_features,
        "--seed", "5",
        "--feat_type", "uni",
        "--tcga_projects", "TCGA-COAD", "TCGA-READ",
        "--start", "0",
        "--end", "109"

    ]
    subprocess.run(cmd, check= True)
    return

def kmeans(ref_file_path, patches, uni_features):
    print("Kmeans")
    cmd= [
        "python", "sequoia-codes/kmean_features.py",
        "--ref_file", ref_file_path,
        "--patch_data_path", patches,
        "--feature_path", uni_features,
        "--seed", "5",
        "--tcga_projects", "TCGA-COAD", "TCGA-READ",
        "--feat_type", "uni",
        "--start", "0",
        "--end", "109"

    ]
    subprocess.run(cmd, check= True)
    return

def rna_pred(ref_file_path, uni_features, results_folder):
    print("Bulk RNA predictions")
    cmd= [
        "python", "sequoia-codes/predict_independent_dataset.py",
        "--ref_file", ref_file_path,
        "--feature_path", uni_features,
        "--seed", "5",
        "--save_dir", results_folder,
        "--tcga_project", "TCGA-COAD", "TCGA-READ",
        "--feature_use", "cluster_features",
        "--exp_name", "sequoia_109"

    ]
    subprocess.run(cmd, check= True)
    return

def rna_corr(info_csv, results, rna_pkl, summary_df, output_path):
    print("Bulk RNA correlations")
    info_df= pd.read_csv(info_csv)
    info_df= info_df.set_index("data_id")
    
    with open(rna_pkl, "rb") as f:
        data = pickle.load(f)
        
    df= pd.DataFrame(data["pred"])
    for index in tqdm(df.index):
        row= df.loc[index]
        row = row.to_frame(name="predicted")
        rna_truth= info_df.loc[index, "rna_files"]
        
        try:
            rna_truth = ast.literal_eval(rna_truth)
            for i, link in enumerate(rna_truth):
                image_code= info_df.loc[index, "patient_barcode"]
                save_dir= os.path.join(results, image_code)
                
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    
                file_name= os.path.join(save_dir, f"{image_code} Spearman correlation {i}.png")
                gt_df= pd.read_csv(link, sep="\t", skiprows=1)
                gt_df = gt_df[["gene_name", "tpm_unstranded"]].dropna(subset=["gene_name"])
                gt_df= gt_df[gt_df["tpm_unstranded"]> 0]
                gt_df= gt_df[gt_df["gene_name"].isin(row.index)].copy()
                gt_df["predicted"]= gt_df["gene_name"].map(row["predicted"])
                gt_df= gt_df.sort_values("gene_name", ascending= True)
                gt_df["tpm_unstranded"] = pd.to_numeric(gt_df["tpm_unstranded"], errors='coerce')
                gt_df["predicted"] = pd.to_numeric(gt_df["predicted"], errors='coerce')
                gt_df.dropna(subset=["tpm_unstranded", "predicted"], inplace=True)
                gt_df["expression_rank"]= gt_df["tpm_unstranded"].rank()
                gt_df["prediction_rank"]= gt_df["predicted"].rank()
                gt_df["expression_log"] = np.log1p(gt_df["tpm_unstranded"])
                gt_df["pred_log"] = np.log1p(gt_df["predicted"])
                gt_df["rank_diff"] = (gt_df["expression_rank"] - gt_df["prediction_rank"]).abs()
                pearson_corr, pearson_p = pearsonr(gt_df["tpm_unstranded"], gt_df["predicted"])
                spearman_corr, spearman_p = spearmanr(gt_df["tpm_unstranded"], gt_df["predicted"])  
                plt.figure(figsize=(6,5))
                sns.regplot(x="expression_rank", y="prediction_rank", data=gt_df,
                            scatter_kws={'s':20}, line_kws={'color':'blue'})
                plt.title(f"{image_code} {i} Spearman correlation: {spearman_corr:.3f}")
                plt.xlabel("Expression Rank")
                plt.ylabel("Prediction Rank")
                plt.tight_layout()
                plt.savefig(file_name)
                plt.close()
                logged_pearson_corr, logged_pearson_p = pearsonr(gt_df["expression_log"], gt_df["pred_log"])
                stats = {
                    "pearson_corr": pearson_corr,
                    "pearson_p": pearson_p,
                    "spearman_corr": spearman_corr,
                    "spearman_p": spearman_p,
                    "logged_pearson_corr": logged_pearson_corr,
                    "logged_pearson_p": logged_pearson_p,
                }        
                gt_df = pd.DataFrame([stats])
                gt_df.to_csv(os.path.join(save_dir, f"{image_code} {i}.csv"), index= False)
                summary_df.loc[f"{image_code}-{i}", ["Bulk RNA Pearson", "Bulk RNA Spearman", "Bulk RNA logged Pearson"]]= [pearson_corr, spearman_corr, logged_pearson_corr]
            
        except Exception as e:
            print(f"Error: {e}")
            print("No ground truth bulk RNA data available")
            continue
            
    file_name= os.path.join(output_path, "summary.csv")
    summary_df.to_csv(file_name)
    return
    
def st_pred(info_csv, results, masks):
    print("ST predictions")
    patch_size= 512
    downscale_factor= 16
    display_spacing= patch_size// downscale_factor
    info_df= pd.read_csv(info_csv)
    gene_names= ["BRAF", "CD68", "EPCAM", "CD3D", "CD3E", "CD3G"]
    
    for row in tqdm(info_df.itertuples(index= False)):
        save_folder= os.path.join(results, row.patient_barcode)
        mask_folder= os.path.basename(row.image_path)
        mask_folder= mask_folder.split(".")[0]
        mask_path= os.path.join(masks, mask_folder, "mask.npy")
        save_name= f"{row.patient_barcode}.csv"
        cmd= [
            "python", "sequoia-codes/visualise_v5.py",
            "--study", "coad",
            "--project", "spatial_coad_pred",
            "--wsi_file_name", row.image_path,
            "--gene_names", ",".join(gene_names),
            "--save_folder", save_folder,
            "--model_type", "vis",
            "--feat_type", "uni",
            "--folds", "0",
            "--checkpoint", results,
            "--patch_size", "256",
            "--stride", "1",
            "--mask", mask_path,
            "--save_name", save_name
    
        ]
        subprocess.run(cmd, check= True)        
        df= pd.read_csv(os.path.join(row.image_path, save_name)).dropna(how= "any")
        df["xcoord"] = df["xcoord"] // downscale_factor
        df["ycoord"] = df["ycoord"] // downscale_factor
        slide, img, w, h, *_= load_slide(slide_path, level= 2)
        
        for gene in gene_names: 
            file_name= os.path.join(save_folder, f"{gene}.png")
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(img)
            sc = ax.scatter(df["xcoord"], df["ycoord"], c=df[gene], cmap="viridis", alpha=1, s=16)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cbar = fig.colorbar(sc, cax=cax)
            cbar.set_label("Predicted Expression")
            ax.set_title(f"{row.patient_barcode} Predicted {gene} Expression Overlay")
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(file_name)
            plt.close()

    return
