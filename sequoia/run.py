from tqdm import tqdm
from utils2 import ref_file_gen, patch_gen, compute_feat, kmeans, rna_pred, rna_corr, st_pred
import os
import argparse
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # pick an unoccupied GPU
    
def main(run_steps, info_csv, output_path):
    ref_file_gen(output_path)
    patches= os.path.join(output_path, "patches")
    masks= os.path.join(output_path, "masks")
    uni_features= os.path.join(output_path, "uni_features")
    results_folder= os.path.join(output_path, "uni_results")
    results= os.path.join(results_folder, "sequoia_109")
    rna_pkl= os.path.join(results, "test_results.pkl")
    summary_df= pd.DataFrame()
    
    if "patch_gen" in run_steps:
        try:
            patch_gen(info_csv, patches, masks)
            
        except Exception as e:
            print(f"Error: {e}")

    if "compute_feat" in run_steps:
        try:
            compute_feat(ref_file_path, patches, uni_features)

        except Exception as e:
            print(f"Error: {e}")

    if "kmeans" in run_steps:
        try:
            kmeans(ref_file_path, patches, uni_features)

        except Exception as e:
            print(f"Error: {e}")

    if "rna_pred" in run_steps:
        try:
            rna_pred(ref_file_path, uni_features, results_folder)

        except Exception as e:
            print(f"Error: {e}")

    if "rna_corr" in run_steps:
        try:
            rna_corr(info_csv, results, rna_pkl, summary_df, output_path)

        except Exception as e:
            print(f"Error: {e}")

    if "st_pred" in run_steps:
        try:
            st_pred(info_csv, results, masks)

        except Exception as e:
            print(f"Error: {e}")        
            
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_steps", nargs="+", default=["patch_gen", "compute_feat", "kmeans", "rna_pred", "rna_corr", "st_pred"],
                        help="Pipeline steps to run, can be run individually")
    parser.add_argument("--info_csv", help= "Path to lookup table")
    parser.add_argument("--output_path", help= "Folder for output files")
    args = parser.parse_args()
    run_steps= args.run_steps
    info_csv= args.info_csv
    output_path= args.output_path
    main(run_steps= args.run_steps, info_csv= args.info_csv, output_path= args.output_path)

# python sequoia-codes/run.py --run_steps st_pred --info_csv /home/jantao/109_info.csv --output_path /home/jantao/sequoia_109
