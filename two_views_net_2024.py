# Side Classifier: 2 views classifier using Sinai Patch Clf for CC, MLO

import sys
import collections
import torch
import torch.nn as nn

from single_full_net_2024 import SingleBreastClassifier
from efficientnet_pytorch import EfficientNet, MBConvBlock

class TwoViewsMIDBreastClassifier(nn.Module):
    """
    Two-views:
    1-Take Feature Extractor part from Full image Classifier (which is the patch classifier part).
    2-Concatenate the outputs (activation maps) and output.
    Topology = MID. Means we concatenate feature extractor from each side and later will stack 
    top layers in the "middle of the way"
    Parameters:
    exp_type = 'PBC', 'DC' : Experiment type, patch based or pure classifier
    """

    def __init__(self, device, model_file, network, exp_type = 'PBC', dataset='CBIS-DDSM'):
        super(TwoViewsMIDBreastClassifier, self).__init__()

        self.single_clf_full = SingleBreastClassifier( network, exp_type, dataset)

        if exp_type == 'PBC':
            self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))      # load patch based model
        elif exp_type == 'DC':
            self.single_clf_full.feature_extractor.load_state_dict(torch.load(model_file, map_location=device))  # load pure model
            extract_layers = 2
            feat_ext = self.feature_extractor = nn.Sequential(*list(self.single_clf_full.feature_extractor.children())[:-extract_layers])
        else:
            # Experiments before 2024
            self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))  

        # take only the weights of 'patch classifier' part, disregard original Top layer(s)
        if exp_type == 'PBC':
            self.single_clf_core = self.single_clf_full.feature_extractor       # PBC & old
        elif exp_type == 'DC':
            self.single_clf_core = feat_ext         # DC only
        else:
            self.single_clf_core = self.single_clf_full.feature_extractor   

    def forward(self, x):
        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        hidden2 = torch.cat([x1_2, x2_2], dim=1)

        return hidden2

class SideMIDBreastModel(nn.Module):
    """ Calls TwoViewsMIDBreastClassifier that instantiates 2-views (CC+MLO) with
        concatenated output (without their original top layer).
        Then append the 2-views top layer, that can be resblocks or MBConv blocks,
        with 1, 2 or 0 count. The last means only FC layer.
        Strides: no. os strides for EficientNets
    """
    connections = 256
    output_size = (4, 2)

    def __init__(self, device, model_file, network, n_blocks, b_type='resnet', avg_pool=True, strides=1, 
                 exp_type = 'PBC', dataset='CBIS-DDSM'):
        super(SideMIDBreastModel, self).__init__()
        if n_blocks not in [0, 1, 2]:
            print('Wrong number of Top Layer blocks.')
            sys.exit()
        self.n_blocks = n_blocks
        # Get two input legs from main classifier:
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network, exp_type, dataset)
        self.avg_pool = avg_pool
        output_channels = 2048

        # 03-Ago-2024
        # Now let's put the top layers above the two legs.
        #  It is based in MBConv blocks based in EfficientNet instances
        #  created here. The width of MBconv should be always fixed
        #  but when is EficientNets we are basing in their size.
        #  For ConvNext we create based in EficientNet-B0 sizes. 
        #  Maybe it should be always like that. Change and text afer 2024-thesis.

        print('Creating Side Mid Networking using:', network, ' and Top Block type: ', b_type, ' Qty: ', n_blocks)
        if 'EfficientNet' in network or 'convnext' in network:

            if b_type == 'mbconv':

                if 'convnext' in network:
                    feat_ext = EfficientNet.from_name('efficientnet-b0', num_classes=2)
                    # global_params = feat_ext._global_params
                    print('Using EFBlocks (top block) parameters from EfficientNet-b0 [Two views creation].')
                    output_channels=inplanes = 2048
                else:
                    # For EfficientNets in 2024
                    feat_ext = EfficientNet.from_name(network.lower(), num_classes=5)  # Experimentos 2views CBIS-2024 estao assim, mudar no futuro para B0 apenas
                
                    if 'b0' in network:
                        inplanes = 2560
                        output_channels=inplanes #1280
                    elif 'b3' in network:
                        inplanes = 3072
                        output_channels=inplanes
                    elif 'b4' in network:
                        inplanes = 3584 #1792
                        output_channels=inplanes #1792 #2048

                self.w_h = 36*28                 # width and height of last layer output

                # Parameters for an individual model block
                BlockArgs = collections.namedtuple('BlockArgs', [
                    'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                    'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

                new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides], expand_ratio=2,
                                    input_filters=inplanes, output_filters=output_channels, se_ratio=0.25, id_skip=True) # para 1 block
  
                # below line for same params for blocks as main net
                block_args, global_params, image_size = new_block, feat_ext._global_params, [15, 15]

                if n_blocks == 1:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size) # comentar para FC-only
                elif n_blocks == 2:
                    self.block1 = MBConvBlock(block_args, global_params, image_size=image_size) # comentar para FC-only
                    new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides], expand_ratio=2,
                                    input_filters=output_channels, output_filters=output_channels, se_ratio=0.25, id_skip=True) # para 1 block
                    self.block2 = MBConvBlock(new_block, global_params, image_size=image_size) # comentar para FC-Only, 1-block

        if self.avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(output_channels, 2)
        else:
            # AVGPOOL
            self.fc_pre = nn.Linear(2048* self.w_h, 1024)  #para aproveitar features espaciais Resnet 9*7 / EficientNet 36*28
            self.fc = nn.Linear(1024, 2)     # INCLUINDO - 2020-10-14 - para aproveitar features espaciais

    def forward(self, x):
        x = self.two_views_clf(x)       # out: torch.Size([1, 4096, 36, 28])
        if self.n_blocks == 1:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
        elif self.n_blocks == 2:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
            x = self.block2(x)          # out: Resnet([1, 2048, 9, 7]) / Eficientnet [2, 2048, 36, 28] 
        if self.avg_pool:
            x = self.avgpool(x)             # out: torch.Size([1, 2048, 1, 1])
            x = torch.flatten(x, 1)         # out: torch.Size([1, 2048])
        else:
            # NO AVGPOOL
            x = x.view(-1, 2048* self.w_h)  # para aproveitar features espaciais  Resnet 9* 7 / Efiencet 36*28
            x = self.fc_pre(x)
        x = self.fc(x)                  # out: torch.Size([1, 2]
        return x
