# Model for Sinai-like full image classifier
from pyexpat import EXPAT_VERSION
import sys
import collections
import torch
import torch.nn as nn

from resnet50 import MyResnet50
# from bottleneck_ import Bottleneck
# from resnet_blocks import ResnetBlocks, Resnet1Blocks

from efficientnet_pytorch import EfficientNet, MBConvBlock

import torchvision.models as models # for 2024 version

# https://github.com/rwightman/pytorch-image-models
from timm import create_model

EXPERIMENT = 2024 # 2024           # 2024 or None (before)

# Loading full pre-trained model instead of patch clf.
# we want to take advanage of pre-trained in CBIS-DDSM full images of course.


class EFBlocks(nn.Module):
    def __init__(self, global_params, image_size=None, inplanes=1792, outplanes=2048, n_blocks=1, strides=1, **kwargs):
        super(EFBlocks, self).__init__()

        #print('Class Param', global_params)

        if EXPERIMENT == 2024:
            temp_model = EfficientNet.from_name('efficientnet-b0', num_classes=5)
            global_params = temp_model._global_params
            print('Using EFBlocks (top block) parameters from EfficientNet-b0 [Single view extraction].')

        #print('Local Param', global_params)


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

        # x = self._conv_head(x)
        # x = self.bn1(x)
        # x = self.swish(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

# Old way to assemble EF blocks - used in CV experiments only
class EFBlocks_cv(nn.Module):
    def __init__(self, block_args, global_params, image_size=None, output_filters=2048, **kwargs):
        super(EFBlocks_cv, self).__init__()
        self.block1 = MBConvBlock(block_args, global_params, image_size=image_size) # comentar para FC-only
        #self.block2 = MBConvBlock(block_args, global_params, image_size=image_size) # comentar para FC-Only, 1-block
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(output_filters, 2)  # 2048 para quando usa block1/ 1792 para FC-only/2-blocks / 1280 para ef-b0


    def forward(self, x):
        # print('Input to block', x.shape)

        x = self.block1(x)  # comentar para FC-only
        # print('Output to block', x.shape)
        #x = self.block2(x)  # comentar para FC-only e 1-block
        # print(x.shape)

        x = self.avgpool(x)
        # print('after avgpool', x.shape)
        x = torch.flatten(x, 1)
        # print('after flatten', x.shape)
        x = self.fc(x)
        # print('after FC', x.shape)
        return x


# Versao correta para Original TEST SPLIT (OD) - 2021-09-07 
# Atualizada para old CV experiments (to load old models) - 2022-05-12

EXP_TYPE = 'OD'   # 'OD'  'CV'

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
    # def __init__(self, network):
        super(SingleBreastClassifier, self).__init__()

        # Para montar MBconv EFBlock 
        #patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # Keep compatibility before 2024

        if network == 'Resnet50':
            # instanciate resnet with 5 outputs
            patch_clf = MyResnet50(num_classes=5)

            # Join Models
            self.feature_extractor = nn.Sequential(
                *list(patch_clf.children())[:-2],     # remove FC layer
            )

            # Get top layers
            # self.top_layer = ResnetBlocks()
            self.top_layer = Resnet1Blocks()

                    #self.top_layer

            # load weights from patch classifier
            # self.patch_clf.load_state_dict(torch.load(model_file, map_location=device))

        elif 'convnext' in network: 

            patch_clf_model = EfficientNet.from_name('efficientnet-b0', num_classes=5)  # Keep compatibility before 2024

            if network == 'convnext_base':   # convnext_base_22k
                model = create_model("convnext_base_in22k", pretrained=False)
                output_filters = inplanes = model.head.fc.in_features
                #model.head.fc = nn.Linear(inplanes, num_classes=2, bias=True)            
                extract_layers = 2
                self.feature_extractor = nn.Sequential(*list(model.children())[:-extract_layers])

            # Trecho abaixo replicando de Efficient Net, depois de ok, deixar generico

            # Parameters for an individual model block
            BlockArgs = collections.namedtuple('BlockArgs', [
                'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

            new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=2,
                                  input_filters=inplanes, output_filters=output_filters, se_ratio=0.25, id_skip=True) # para 1 block
                                # input_filters=inplanes, output_filters=1792, se_ratio=0.25, id_skip=True) # para FC-only, 2-blocks
                                # input_filters=inplanes, output_filters=1280, se_ratio=0.25, id_skip=True) # PARA EFICIENT-B0

            # print(new_block)

            # Select below for Efficient Net topo layer
            if EXP_TYPE == 'CV':
                self.top_layer = EFBlocks_cv(new_block, patch_clf_model._global_params, [15, 15], output_filters=output_filters)
            elif EXP_TYPE == 'OD':
                self.top_layer = EFBlocks(patch_clf_model._global_params, [15, 15], inplanes=inplanes,
                                          outplanes=output_filters, n_blocks=2, strides=2)


        elif 'EfficientNet' in network:   

            # linha de baixo comentada em 03-Ago-2024 para ser maior escopo para montar EFBlock
            patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # Keep compatibility before 2024

            # if EXPERIMENT == 2024:
            #     print('-> Experiment 2024')     

            if exp_type == 'PBC' and dataset in ['CBIS-DDSM', 'VINDR_MAMMO']:
                # PBC - Fixed classifier to test, from thesis text
                print('-> Experiment 2024: ', exp_type, dataset) 
                model = models.efficientnet_b3(weights=None, num_classes=2)
                output_filters = inplanes = model.classifier[1].in_features
                extract_layers = 2
                self.feature_extractor = nn.Sequential(*list(model.children())[:-extract_layers])
            elif exp_type == 'DC' and dataset in ['CBIS-DDSM', 'VINDR_MAMMO']:
                # DC
                print('-> Experiment 2024: ', exp_type, dataset)
                self.feature_extractor = models.efficientnet_b3(weights=None, num_classes=2)
                output_filters = inplanes = self.feature_extractor.classifier[1].in_features
            else:
                # Experiments before 2024
                # patch_clf_model = EfficientNet.from_name(network.lower(), num_classes=5)  # See Coment in definition (my change)
                self.feature_extractor = patch_clf_model

            #model_file = load_patch_model(k, 'EfficientNet-b4')
            # patch_clf_model.load_state_dict(torch.load(model_file, map_location=device))
            

            if 'b0' in network:
                output_filters = 1280
                inplanes = 1280

            elif 'b4' in network:
                output_filters = 2048 #1792  #2048
                inplanes = 1792

            # self.feature_extractor = patch_clf_model

            # Parameters for an individual model block
            BlockArgs = collections.namedtuple('BlockArgs', [
                'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
                'input_filters', 'output_filters', 'se_ratio', 'id_skip'])

            new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=2,
                                  input_filters=inplanes, output_filters=output_filters, se_ratio=0.25, id_skip=True) # para 1 block
                                # input_filters=inplanes, output_filters=1792, se_ratio=0.25, id_skip=True) # para FC-only, 2-blocks
                                # input_filters=inplanes, output_filters=1280, se_ratio=0.25, id_skip=True) # PARA EFICIENT-B0

            # print(new_block)

            # Select below for Efficient Net topo layer
            if EXP_TYPE == 'CV':
                self.top_layer = EFBlocks_cv(new_block, patch_clf_model._global_params, [15, 15], output_filters=output_filters)
            elif EXP_TYPE == 'OD':
                self.top_layer = EFBlocks(patch_clf_model._global_params, [15, 15], inplanes=inplanes,
                                          outplanes=output_filters, n_blocks=2, strides=2)

        else:
            print('Wrong selection of network')


    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.top_layer(x)

        return x





# # VErsao anterior usada sem CV, antes eficientNET (versao OLD)
# class SingleBreastClassifier(nn.Module):

#     def __init__(self, device, model_file):
#         super(SingleBreastClassifier, self).__init__()

#         # instanciate resnet with 5 outputs
#         self.patch_clf = MyResnet50(num_classes=5)

#         # Get top layers
#         self.top_layer = ResnetBlocks()

#         # Join Models
#         self.net = nn.Sequential(
#             *list(self.patch_clf.children())[:-2],     # remove FC layer
#             self.top_layer
#         )

#         # load weights from patch classifier
#         self.net.load_state_dict(torch.load(model_file, map_location=device))


#     def forward(self, x):

#         x = self.net(x)

#         return x


# Nao da certo devido que nao queremos carregar os pesos do patch clf 
# e sim do Full CLF, como feito acima.
# class SingleBreastClassifier(nn.Module):

# 	def __init__(self, device, model_file):
# 		super(SingleBreastClassifier, self).__init__()

# 		# instanciate resnet with 5 outputs
# 		self.patch_clf = MyResnet50(num_classes=5)

# 		# load weights from patch classifier
# 		self.patch_clf.load_state_dict(torch.load(model_file, map_location=device))

# 		# Get top layers
# 		self.top_layer = ResnetBlocks()

# 		# Join Models
# 		self.net = nn.Sequential(
# 			*list(self.patch_clf.children())[:-2],     # remove FC layer
# 			self.top_layer
# 		)

# 	def forward(self, x):

# 		x = self.net(x)

# 		return x


# jeito bonito, que separa as layers mas depois nao bate os nomes dos
# parametros. Precisaria fazer assim desde o treino no full image.
# class SingleBreastClassifier(nn.Module):

#     def __init__(self, device, model_file):
#         super(SingleBreastClassifier, self).__init__()

#         # instanciate resnet with 5 outputs
#         self.patch_clf = MyResnet50(num_classes=5)

#         # load weights from patch classifier
#         self.patch_clf.load_state_dict(torch.load(model_file, map_location=device))

#         # Get top layers
#         self.top_layer = ResnetBlocks()

#         # Remove avgpool and fc from resnet50
#         self.patch_clf_core = nn.Sequential(
#             *list(self.patch_clf.children())[:-2],     # remove FC layer
#         )

#     def forward(self, x):

#         x = self.patch_clf_core(x)
#         x = self.top_layer(x)

#         return x
