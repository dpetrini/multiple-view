# Multiple-view
Multiple view inference models for breast cancer detection.
This is the inference code of the Multiple View classifiers to classify the two mammography views at once. It was trained in CBIS-DDSM dataset and in VinDr-Mammo with original test split. It means that any pair of mammograms in test set can be used in this inference.

## Instructions for inference with multiple views

python3 multi_view_clf_test.py -h

```
usage: multi_view_clf_test.py [-h] -c CC -m MLO [-d MODEL] [-a AUG]

[Poli-USP] Multiple Views Breast Cancer inference

optional arguments:
  -h, --help                    show this help message and exit
  -c CC, --cc CC                CC image file.
  -m MLO, --mlo MLO             MLO image file.
  -t TYPE, --type TYPE          two-views paradigm PBC or DC
  -d DATASET, --data DATASET    select DATASET (CBIS-DDSM or VINDR-MAMMO) 
  
```

  Example:
```
  python3 multi_view_clf_test.py  -c samples/Calc-Test_P_00041_LEFT_CC.png -m samples/Calc-Test_P_00041_LEFT_MLO.png
```
Obs. Some sample files from CBIS-DDSM test set are included in samples folder for evaluation. Files were resized for network input.

Obs2. In order to perform test inference download our muliple-views model from [link] and place it in "models" folder.

### Dependencies
argparse

numpy

torch

cv2


### Reference
If you use want to know more, please check complete text HERE. If you want to cite this work please use reference below.

```
@ARTICLE{
}
