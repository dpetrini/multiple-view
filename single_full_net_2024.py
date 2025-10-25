# Model for Sinai-like full image classifier
# from pyexpat import EXPAT_VERSION
import sys
import collections
import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet, MBConvBlock

import torchvision.models as models # for 2024 version

# https://github.com/rwightman/pytorch-image-models
from timm import create_model

# Loading full pre-trained model instead of patch clf.
# we want to take advanage of pre-trained in CBIS-DDSM full images of course.

class EFBlocks(nn.Module):
    def __init__(self, global_params, image_size=None, inplanes=1792, outplanes=2048, n_blocks=1, strides=1, **kwargs):
        super(EFBlocks, self).__init__()

        temp_model = EfficientNet.from_name('efficientnet-b0', num_classes=5)
        global_params = temp_model._global_params
        print('Using EFBlocks (top block) parameters from EfficientNet-b0 [Single view extraction].')

        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon
        self.drop_connect_rate = global_params.drop_connect_rate
        self.n_blocks = n_blocks

        # Parameters for an individual model block
        BlockArgs = collections.namedtuple('BlockArgs', [
            'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
            'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

        block_args = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides],
                               expand_ratio=2, input_filters=inplanes,
                               output_filters=outplanes, se_ratio=0.25, id_skip=True)

        self._blocks = nn.ModuleList([])
        for _ in range(0, n_blocks):
            self._blocks.append(MBConvBlock(block_args, global_params, image_size=image_size))

        # Conv2d = get_same_padding_conv2d(image_size=image_size)
        # self._conv_head = Conv2d(outplanes, outplanes, kernel_size=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(num_features=outplanes, momentum=bn_mom, eps=bn_eps)
        # self.swish = MemoryEfficientSwish()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(global_params.dropout_rate/2)
        self.fc = nn.Linear(outplanes, 2)

    def forward(self, x):
        # print('Input to block', x.shape)

        for i, block in enumerate(self._blocks):
            drop_connect_rate = self.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(i) / len(self._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate)
            # print('Output to block', i+1, x.shape, drop_connect_rate)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

class SingleBreastClassifier(nn.Module):
    """
    SingleBreast Classifier is the Full image classifier.
    In this class we reproduce the single-view model to separate in two parts:
    Feature extractor => that is the patch classifier
    Top Layer => the top blocks. [As for the AACR abstracts no. of blocks=1 ]
    Usually we will use only feature extractor for 2-views classifier.

    Here we only assemble model, we dont load specific weights here.
    """

    def __init__(self, network, exp_type = 'PBC', dataset='CBIS-DDSM'):
        super(SingleBreastClassifier, self).__init__()

        if 'convnext' in network: 

            patch_clf_model = EfficientNet.from_name('efficientnet-b0', num_classes=5)  # Keep compatibility before 2024

            if network == 'convnext_base':   # convnext_base_22k
                model = create_model("convnext_base.fb_in22k", pretrained=False)
                output_filters = inplanes = model.head.fc.in_features
                #model.head.fc = nn.Linear(inplanes, num_classes=2, bias=True)            
                extract_layers = 2
                self.feature_extractor = nn.Sequential(*list(model.children())[:-extract_layers])

            self.top_layer = EFBlocks(patch_clf_model._global_params, [15, 15], inplanes=inplanes,
                                          outplanes=output_filters, n_blocks=2, strides=2)

        elif 'EfficientNet' in network:   

            # linha de baixo comentada em 03-Ago-2024 para ser maior escopo para montar EFBlock
            patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # Keep compatibility before 2024  

            if exp_type == 'PBC' and dataset in ['CBIS-DDSM', 'VINDR-MAMMO']:
                # PBC - Fixed classifier to test, from thesis text
                print('-> Experiment 2024: ', exp_type, dataset) 
                model = models.efficientnet_b3(weights=None, num_classes=2)
                output_filters = inplanes = model.classifier[1].in_features
                extract_layers = 2
                self.feature_extractor = nn.Sequential(*list(model.children())[:-extract_layers])
            elif exp_type == 'DC' and dataset in ['CBIS-DDSM', 'VINDR-MAMMO']:
                # DC
                print('-> Experiment 2024: ', exp_type, dataset)
                self.feature_extractor = models.efficientnet_b3(weights=None, num_classes=2)
                output_filters = inplanes = self.feature_extractor.classifier[1].in_features
            else:
                # Experiments before 2024
                # patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # See Coment in definition (my change)
                self.feature_extractor = patch_clf_model

            if 'b0' in network:
                output_filters = 1280
                inplanes = 1280

            elif 'b4' in network:
                output_filters = 2048 #1792  #2048
                inplanes = 1792

            self.top_layer = EFBlocks(patch_clf_model._global_params, [15, 15], inplanes=inplanes,
                                          outplanes=output_filters, n_blocks=2, strides=2)

        else:
            print('Wrong selection of network')


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.top_layer(x)

        return x

