import os
import pandas as pd
import openslide

csv_path= "/home/jantao/TCGA_Validation_109WSIs.csv"
rna_path= "/data_g1/AI_projects/raw_data/TCGA_CRC/Transcriptome_Profiling/tcga_query_results.csv"
cell_path= "/data_g1/AI_projects/raw_data/TCGA_CRC/tcga_crc_metadata.csv"
df= pd.read_csv(csv_path)
rna_df= pd.read_csv(rna_path)
cell_df= pd.read_csv(cell_path)
df= df[["data_id", "patient_barcode", "image_path"]]
cancer_types= []
rna_paths= {}

for _, row in df.iterrows():
    parts= row["image_path"].split(os.sep)
    cancer_type= parts[-3]
    cancer_type = cancer_type.replace("_", "-")
    cancer_types.append(cancer_type)
    image_name= row["patient_barcode"]
    matches = rna_df[rna_df["cases.submitter_id"] == image_name]
    
    for _, row in matches.iterrows():
        file_name= row["file_name"]
        folder_id= row["id"]

        if image_name not in rna_paths:
            rna_paths[image_name] = []
        path= os.path.join("/data_g1/AI_projects/raw_data/TCGA_CRC/Transcriptome_Profiling/Gene_Expression_Quantification/", cancer_type, folder_id, file_name)
        rna_paths[image_name].append(path)
        
df["cancer_type"]= cancer_types
df["rna_files"] = df["patient_barcode"].map(rna_paths)
df = df.merge(
    cell_df[["data_id", "cell_data_path"]],
    on="data_id",
    how="left"   # keep all rows in df, even if missing in cell_df
)
df.to_csv("/home/jantao/109_info.csv")
