# Multiple-view
Multiple view inference models for breast cancer detection.
This is the inference code of the Multiple View classifiers to classify the two mammography views at once. It was trained in CBIS-DDSM dataset and in VinDr-Mammo with original test split. It means that any pair of mammograms in test set can be used in this inference.

## Instructions for inference with multiple views

python3 multi_view_clf_test.py -h

```
usage: multi_view_clf_test.py [-h] -c CC -m MLO [-d MODEL] [-a AUG]

[Poli-USP] Multiple Views Breast Cancer inference

optional arguments:
  -h, --help               show this help message and exit
  -c CC, --cc CC           CC image file.
  -m MLO, --mlo MLO        MLO image file.
  -d MODEL, --model MODEL  two-views detector model (default model already included)
  -a AUG, --aug AUG        select to use translation augmentation: -a true
  
```

  Example:
```
  python3 multi_view_clf_test.py -c samples/Calc-Test_P_00127_RIGHT_CC.png -m samples/Calc-Test_P_00127_RIGHT_MLO.png
```
Obs. Some sample files from CBIS-DDSM test set are included in samples folder for evaluation. Files were resized for network input.

Obs2. In order to perform test inference download our muliple-views model from [] and place it in "models_side_mid_clf_efficientnet-b3" folder.

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
