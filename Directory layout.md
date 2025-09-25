# Directory layout

Here is the description of my directory, in alphabetical order. Important files/folders are in **bold**
<br>
<br>

**109\_info.csv**

The lookup table for file paths and names for TCGA\_Validation\_190WSIs. Created using info\_generation.py, which can be found in the sequoia folder in this repo
<br>
<br>

bulk\_rna\_data and cell\_tables

Bulk RNA and cell distribution ground truth for selected images
<br>
<br>

CHIEF

Cloned directly from https://github.com/hms-dbmi/CHIEF. Not many changes made, but they can be found in the py files at the top level with "edited" in the name. Reported during the 4/9 meeting
<br>
<br>

chief-env

The virtual environment set up for CHIEF
<br>
<br>

**DeepSpot**

Cloned directly from https://github.com/ratschlab/DeepSpot. Files and folders created subsequently are all at the top level, and include: test\_data, run\_step2.py, run.py, the whole package.ipynb and utils.py

test\_data: Prediction data for selected images

run\_step2.py, run.py and utils.py: Used to run images at bulk. Also published on the deepspot folder in this repo. If the files on GitHub fail to run, use these files in this folder

the whole package.ipynb: An earlier version of run\_step2.py, run.py and utils.py, in notebook format
<br>
<br>

**deepspot\_109**

DeepSpot prediction data of all 109 images
<br>
<br>

**deepspot\_env**

The virtual environment set up for DeepSpot
<br>
<br>

edited-files

Outdated, edited SEQUOIA code files. Created when moving from personal laptop to SIgN desktop. For up-to-date version see sequoia-codes
<br>
<br>

HistoBistro

Cloned directly from https://github.com/peng-lab/HistoBistro. No changes made. Reported during 4/9 meeting
<br>
<br>

images

The images shared through Google Drive
<br>
<br>

istar and Loki

Cloned directly from https://github.com/daviddaiweizhang/istar/tree/master and https://github.com/GuangyuWangLab2021/Loki respectively. loki\_predex can be downloaded from the Google Drive link provided by https://guangyuwanglab2021.github.io/Loki/notebooks/Loki\_PredEx\_case\_study.html. Reported during 7/8 meeting
<br>
<br>

istar-env and loki-env

The virtual environment set up for istar
<br>
<br>

marugoto

Cloned directly from https://github.com/KatherLab/marugoto. Failed to run due to dependency issues. Reported during 7/8, 28/8 and 4/9 meetings

marugoto/mil/data.py- some fixes attempted, as denoted by ###

clini\_table.csv and slide\_csv.csv- simple table needed to run their code

table\_generation.ipynb- generates clini\_table.csv and slide\_csv.csv, and also failed implementation of model on our images (model provided incompatible)

Various .pth, .pkl and .ckpt files- models provided by Kather lab:

best\_ckpt.pth can be downloaded from the Google Drive link found in the readme in the marugoto repo

Exp\_MODEL\_Full from https://zenodo.org/records/5151502

export-0.pkl from https://github.com/KatherLab/crc-models-2022/blob/main/Quasar\_models/Wang%2BattMIL%2Btabular/isMSIH/export-0.pkl

fastai\_weights.pth and my\_model.ckpt are failed file conversions from export-0.pkl. See table\_generation.ipynb for code

MSI\_high\_CRC\_model.pth: unfortunately I do not recall where this is from
<br>
<br>

marugoto-outputs

Some preprocessing outputs produced before I realised Marugoto does not predict weights for their final prediction module

Folders in tiles/BLOCKS are the same as the TCGA folders a level above
<br>
<br>

**MSI\_prediction**

Folder for work done to predict MSI status

conch_patches: Contains patches and features extracted by TRIDENT. Also some h5 files containing aggregated CONCH features, the direct input for prediction modules

uni_masks: masks for all images in the entire Harvard dataset, including those that are not included in the analysis

better_patches.h5: UNI patches extracted after the masking protocol

better_uni_features.h5: Features extracted from better_patches.h5

conch.ipynb: Prediction workflow using CONCH

features.h5: UNI features extracted from patches.h5

mean_aggregated_better_uni_features.h5: the aggregated version of better_uni_features.h5

mean_aggregated_features.h5: the aggregated version of features.h5

patches.h5: UNI patches extracted without using the masking protocol

preprocessed_data.pkl: Contains info for dataset cohort. Which images/ patients should belong in either training or validation sets of each fold, or if it should be in hold-out test set. Used only with the CONCH workflow

preprocessing.py: Used to run feature extraction overnight for UNI. Likely to be outdated, do not recommend using

uni.ipynb: prediction workflow using UNI
<br>
<br>

not\_marugoto\_env

The environment set up for Marugoto
<br>
<br>

preprocessing and preprocessing-ng

Preprocessing steps as cloned from https://github.com/KatherLab/preProcessing and https://github.com/KatherLab/preprocessing-ng. They are used in early publications by Kather lab, and other papers inspired by Kather et al.
<br>
<br>

Various Python files

Different Python versions to run various virtual environments
<br>
<br>

ref\_file.csv

csv file example provided by SEQUOIA required to run their workflow. Downloaded from https://github.com/gevaertlab/sequoia-pub/blob/master/examples/ref\_file.csv. Also see sequoia-codes/move/convert to ref\_file.ipynb
<br>
<br>

requirements.txt

I do not remember what this is for
<br>
<br>

**sequoia\_109**

SEQUOIA prediction data of all 109 images
<br>
<br>

**sequoia-codes**

Cloned from https://github.com/gevaertlab/sequoia-pub and some of them are subsequently modified. Can also be found in the sequoia folder in this repo. model.safetensors is downloaded from https://huggingface.co/gevaertlab/sequoia-coad-0
<br>
<br>

**sequoia-codes/move**

also patch visualisation.ipynb: Visualise patches and masks produced by SEQUOIA

convert to ref\_file.ipynb: Code used to generate ref\_file.csv

czi file.ipynb: Code used to visualize liver cancer images in czi format

Various safetensors files: Downloaded from https://huggingface.co/gevaertlab/models. They are either for liver or colorectal cancer

miscellaneous.ipynb: Also found in the sequoia folder of this repo. See the readme in the same folder

patch visualisation backup.ipynb and patch visualisation.ipynb: inspection of features and other intermediate files produced by the SEQUOIA workflow. Also some early bulk RNA analysis

patch\_gen\_hdf5.py: An alternate version of the same file found one level above

requirements.txt: requirements for sequoia-env, with some edits made

Untitled.ipynb: Full version can be found in sequoia-outputs

Untitled2.ipynb: Never used, can ignore
<br>
<br>

sequoia-env

The virtual environment set up for SEQUOIA
<br>
<br>

**sequoia-outputs**

Contains all outputs on test images and ref\_files for each set of test images as generated by sequoia-codes/move/convert to ref\_file.ipynb
<br>
<br>

**sequoia-outputs/uni\_visualisation**

Contains ST prediction of various test images of various patch sizes and models

sequoia-outputs/uni\_visualisation/For Jan: Cell type distribution 'ground truth'

cells.ipynb: Visualise cell distribution predictions and statistical comparisons

heatmap.ipynb: Visualise ST predictions
<br>
<br>

STAMP

Cloned from https://github.com/KatherLab/STAMP
<br>
<br>

TCGA\_Validation\_109WSIs.csv

Ground truth info 
<br>
<br>

trident

Cloned from https://github.com/mahmoodlab/TRIDENT. Used to predict MSI status
<br>
<br>

trident-env

The virtual environment set up for TRIDENT
<br>
<br>

UNI

Cloned from https://github.com/mahmoodlab/UNI
<br>
<br>

