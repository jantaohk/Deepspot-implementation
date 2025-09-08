import pandas as pd
import openslide

def load_slide(slide_path, level=2):
    slide = openslide.OpenSlide(slide_path)
    w, h = slide.level_dimensions[level]
    img = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    mpp_x = float(slide.properties.get("openslide.mpp-x", "nan"))
    mpp_y = float(slide.properties.get("openslide.mpp-y", "nan"))
    #plt.figure()
    #plt.imshow(img)
    #plt.axis("off")
    #plt.show()
    
    return slide, img, w, h, mpp_x, mpp_y
    
def get_image_path():
    image_paths= []
    image_codes= []
    csv_path= "/home/jantao/TCGA_Validation_109WSIs.csv"
    df= pd.read_csv(csv_path)
    
    for index, row in df.iterrows():
        image_paths.append(row["image_path"])
        name= row["patient_barcode"]
        image_code = "-".join(name.split("-")[1:3])
        image_codes.append(image_code)
        
    return image_paths, image_codes

def convert_cell_table(info_df, image_code, slide_path):
    *_, mpp_x, mpp_y= load_slide(slide_path, level= 2)
    edited_image_code= "TCGA-"+ image_code
    cell= info_df.loc[info_df["patient_barcode"]== edited_image_code, "cell_data_path"]

    if cell.empty:
        print(f"No cell data found for {image_code}. Skipping.")
        return
        
    elif len(cell)> 1:
        print(f"Multiple cell data files found for {image_code}. Skipping.")
        return

    else:
        cell_path= cell.iloc[0]
        df = pd.read_csv(cell_path, sep="\t", encoding="latin-1", header=0, engine="python")
        df.columns = df.columns.str.strip()
        df = df.map(lambda v: v.strip() if isinstance(v, str) else v)
            
        for col in ("Centroid X µm", "Centroid Y µm"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                    
        df= df.rename(columns= {"Name": "cell_label"})
        df= df.drop(["Class", "ROI"], axis= 1)
        df["Centroid X µm"]= df["Centroid X µm"]/ mpp_x
        df["Centroid Y µm"]= df["Centroid Y µm"]/ mpp_y
        df= df.rename(columns= {"Centroid X µm": "x", "Centroid Y µm": "y"})
        df.to_csv(f"/home/jantao/cell_tables/{image_code}_cell_table.csv")
        
    return df
