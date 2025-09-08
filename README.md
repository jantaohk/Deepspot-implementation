# Deepspot-implementation
This project predicts spatial transcriptomics from H&E images.  
Following instructions from [here](https://github.com/ratschlab/DeepSpot/blob/main/README.md), first set up a virtual environment for DeepSpot.  
Next install UNI from [here](https://github.com/mahmoodlab/UNI/blob/main/README.md), pick the version from March 2024.  
Then run run.py to make predictions  
Followed by info_generation.py to create a lookup table for evaluation  
And finally run_step2.py  
There are also example bash codes at the bottom of each .py file.

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
