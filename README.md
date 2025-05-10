# STELLA - Segmentation Tool for Enhanced Localisation and Labelling of diagnostic Areas
This repository is the official code for the paper "Toward Paediatric Digital Twins: STELLA-Segmentation Tool for Enhanced Localisation and Labelling of diagnostic Areas" by Roberta De Fazio, Maria Stella de Biase, Pierluigi Marzuillo, Paola Tirelli, Fiammetta Marulli, Stefano Marrone, Laura Verde.
![Workflow](https://github.com/Ste-lla02/stella/blob/roberta/Figures/stella_pipeline.pdf)
## Citation
Please cite our work if you find it useful for your research and work.

```
@article{DeFazio2024,
  title = {Toward Paediatric Digital Twins: STELLA-Segmentation Tool for Enhanced Localisation and Labelling of diagnostic Areas},
  volume = {},
  ISSN = {},
  DOI = {},
  journal = {Procedia Computer Science},
  publisher = {Elsevier BV},
  author = {De Fazio,  Roberta and de Biase,  Maria Stella and Marzuillo,  Pierluigi and Tirelli, Paola and Marulli, Fiammetta and Marrone,  Stefano and Verde,  Laura},
  year = {2025},
  pages = {}
}
```

## Dependencies

The code relies on the following Python 3.9 + libs.

Packages needed are:
* torch 2.6.0
* torchvision 0.21.0
* matplotlib 3.9.4
* pandas 2.2.3
* numpy 1.26.4
* opencv-python 4.11.0.86
* ipython 8.15.0
* scikit-learn 1.6.1
* Pillow 11.1.0
* tensorflow 2.19.0
* keras 3.9.2
* pytorch-cuda 11.8 
* seaborn 0.13.2

  
### Download and Integration of the SAM2 Model

To download and set up the SAM2 model, follow these steps:

```bash
git clone https://github.com/facebookresearch/segment-anything.git

wget -O models/sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Execution
### Parameters configuration in configuration.ini
### Folder Structure Setup
To initialize the required folder structure for the project, run the following commands in your terminal:

```bash
mkdir -p experiments/img/dumps experiments/img/test experiments/img/preprocessed experiments/img/masks experiments/input experiments/output/models experiments/output/report_labels experiments/models
```
### Preprocessing & Mask Generation
To run preprocessing and SAM models on the image:
```bash
python3 main.py configuration.ini classification
```
### Labelling
To run Dobby tool for masks labelling:
```bash
python3 main.py configuration.ini process
```

### Augmentation & Classification
Remember to set "augmentation" in section [classification] under keyword "preprocessing:" in configuration.ini

To run Augmentation and masks Classification:
```bash
python3 main.py configuration.ini classification
```
