# Deepspot-implementation
This project predicts spatial transcriptomics from H&E images.  
For SEQUOIA instructions please see the readme in that folder.  
Following instructions from [here](https://github.com/ratschlab/DeepSpot/blob/main/README.md), first set up a virtual environment for DeepSpot.  
Next install UNI from [here](https://github.com/mahmoodlab/UNI/blob/main/README.md), pick the version from March 2024.  
Download the relevant models from [here](https://zenodo.org/records/15322099) and store them in DeepSpot/DeepSpot_pretrained_model_weights. There should be 3 files to download for each model, the pkl, csv and yaml files. Change the file paths (denoted by #####) in run.py to link to those files.  
Then run run.py to make predictions. This is adapted directly from DeepSpot's example notebook [here](https://github.com/ratschlab/DeepSpot/blob/main/example_notebook/Visium_spot_example/GettingStartedWithDeepSpot_3.1_inference_pretrained_models.ipynb).  
Followed by info_generation.py to create a lookup table for evaluation. Change the file path in run_step2.py (denoted by #####) so it links to the lookup table.  
And finally run_step2.py. More info regarding what each function and parser argument does can be found in the file itself.  
There are also example bash codes at the bottom of each important .py file.

## Acknowledgements / Citations

This repository builds upon code and models provided by:

Nonchev et al. (2025), *DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H&E Images*, medRxiv.  

```bibtex
@article{nonchev2025deepspot,
  title={DeepSpot: Leveraging Spatial Context for Enhanced Spatial Transcriptomics Prediction from H\&E Images},
  author={Nonchev, Kalin and Dawo, Sebastian and Silina, Karina and Moch, Holger and Andani, Sonali and Tumor Profiler Consortium and Koelzer, Viktor H and Raetsch, Gunnar},
  journal={medRxiv},
  pages={2025--02},
  year={2025},
  publisher={Cold Spring Harbor Laboratory Press}
}
