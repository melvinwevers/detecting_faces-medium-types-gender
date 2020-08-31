# Detecting Faces, Medium Types,and Gender in Historical Advertisements 

This repo belongs to paper presented at VISART V workshop at ECCV2020.


## Abstract

Libraries, museums, and other heritage institutions are digitizing large parts of their archives. Computer vision techniques enable scholars to query, analyze, and enrich the visual sources in these archives. However, it remains unclear how well algorithms trained on modern photographs perform on historical material. This study evaluates and adapts existing algorithms. We show that we can detect faces, visual media types, and gender with high accuracy in historical advertisements. It remains difficult to detect gender when faces are either of low quality or relatively small or large. Further optimization of scaling might solve the latter issue, while the former might be ameliorated using upscaling. We show how computer vision can produce meta-data information, which can enrich historical collections. This information can be used for further analysis of the historical representation of gender.


## Instructions

Take files from Zenodo 10.5281/zenodo.4008991

- Place `annotations.tar.gz` in `data/raw`
- Place `ads_meta.csv.zip` (metadata information) in `data/processed`
- place `gt_faces.zip` (ground-truth) in `data/processed`
- Place `dnn_detections.zip` (openCV face detection output) in `data/processed`
- Place `medium classifier_training.zip` files in `data/processed`
- Place `gender_faces.zip` in `data/processed`
- Place `models.zip` in `models`

Also place SIAMESET collection of ads in this folder. This set can be acquired from the National Library of the Netherlands.

## Project Organization

### Data

This folder contains the annotations and training material for the classifiers

### Detecting Gender

This folder contains contains the code for the training of the gender classifier
`fine_tune_2step.py` is the script to finetune the model for classes `male` and `female`

`fine_tune_multiclass.py` allows for for training of gender and medium type classes

### FaceDetection-DSFD

Contains the adapted DSFD repo 

### Models

Outputted models for gender and medium type detection

### Notebooks

`0-mw-preparing_meta_data.ipynb` describes the preparation of the meta data info and the training files

`1-mw-analysis_meta_data.ipynb` analyses metadata

`2-mw-make_gt_dt_dnn.ipynb` shows how to produce ground truth from annotations

`3-mw-separatephoto.ipynb` training of the medium classifier

`4-mw-inspecting_gender_prediction_model.ipynb` inspecting of the gender classifier training in `detecting_gender` folder

### Pytorch_Retinaface

contains the adapted retinaface repo

### Reports

Contains results of face detection, figures used in paper, and data underlying figures. 

### src

helper scripts for cropping, sampling, erasing, etc..
