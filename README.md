# Multiple-view
Multiple view inference models for breast cancer detection.
This is the inference code of the Multiple View classifiers to classify the two mammography views at once. It was trained in CBIS-DDSM dataset and in VinDr-Mammo with original test split. It means that any pair of mammograms in test set can be used in this inference.

## Update: new source code inserted 

Dataloaders for single-view and two-view. New in folder Dataloader.

Keep watching this repository as more code will be added soon.
If you have any inquire or request please use the issues tab.

Also, check the library we use for training the models: [https://github.com/dpetrini/nova](https://github.com/dpetrini/nova)


## Instructions for inference with multiple views

python3 multi_view_clf_test.py -h

```
usage: multi_view_clf_test.py [-h] -c CC -m MLO [-d MODEL] [-a AUG]

[Poli-USP] Multiple Views Breast Cancer inference

optional arguments:
  -h, --help                    show this help message and exit
  -c CC, --cc CC                CC image file.
  -m MLO, --mlo MLO             MLO image file.
  -d DATASET, --data DATASET    select DATASET, CBIS-DDSM (default) or VINDR-MAMMO.
  
```

  Example:
```
  python3 multi_view_clf_test.py  -c samples/Calc-Test_P_00127_RIGHT_CC.png -m samples/Calc-Test_P_00127_RIGHT_MLO.png
```
Obs. Some sample files from CBIS-DDSM test set are included in samples folder for evaluation. Files were resized for network input. The results for ""...127..." images should be 0.1214 if selected CBIS-DDSM or 0.7873, if selectec VINDR-MAMMO.

Obs2. In order to perform test inference download all models from [link](https://drive.google.com/drive/folders/1aQqX2F5f62D2GZOeZmGrmPvwLy_tG9Tr?usp=sharing) and place it in "models" folder.

New: the multiple-views models for CBIS-DDSM and VinDr-Mammo are now available at [Hugging Face](https://huggingface.co/dpetrini). Download and follow the instructions above.

### Dependencies
argparse

numpy

torch

cv2

timm


### Reference
If you use want to know more, please check complete text [HERE](https://arxiv.org/abs/2503.19945). If you want to cite this work please use reference below.

```
@misc{petrini2025optimizingbreastcancerdetection,
      title={Optimizing Breast Cancer Detection in Mammograms: A Comprehensive Study of Transfer Learning, Resolution Reduction, and Multi-View Classification}, 
      author={Daniel G. P. Petrini and Hae Yong Kim},
      year={2025},
      eprint={2503.19945},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2503.19945}, 
}
