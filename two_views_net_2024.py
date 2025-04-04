# Side Classifier: 2 views classifier using Sinai Patch Clf for CC, MLO

import sys
import collections
import torch
import torch.nn as nn

# from constants import SIDES
# from resnet50 import MyResnet50
# from bottleneck_ import Bottleneck
from single_full_net_2024 import SingleBreastClassifier

from efficientnet_pytorch import EfficientNet, MBConvBlock

# patch clf weights
MODEL_PATCH_CLF = 'models_train/2020-06-16-08_32_48_50ep_44686n_best_model__S10_BEST_Val_097_test_097_50ep.pt' # S10
MODEL_FULL_TRAINED = 'models_train/2020-10-02-11h46m_50ep_best_model_BS5_AUC_09605_CBIS-DDSM_Best.pt'
MODEL_ICESP_FULL = 'models_train/2020-08-16-21h02m_25ep_best_model_AUC_09514.pt'

MODEL_FULL_CLF = MODEL_ICESP_FULL


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

        # self.single_clf_full = SingleBreastClassifier(device, MODEL_FULL_CLF)
        self.single_clf_full = SingleBreastClassifier( network, exp_type, dataset)

        # print(self.single_clf_full)

        # # print(model)
        # cont = 0
        # for name, param in self.single_clf_full.named_parameters():
        #     print(cont, name)
        #     cont+=1
        # import sys
        # sys.exit()


        if exp_type == 'PBC':
            self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))      # load patch based model
        elif exp_type == 'DC':
            self.single_clf_full.feature_extractor.load_state_dict(torch.load(model_file, map_location=device))  # load pure model
            extract_layers = 2
            feat_ext = self.feature_extractor = nn.Sequential(*list(self.single_clf_full.feature_extractor.children())[:-extract_layers])
        else:
            # Experiments before 2024
            self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))  


        # retrieve only patch clf, leaves behind resblocks, avgpool,FC
        # self.single_clf_core = nn.Sequential(
        #     *list(list(self.single_clf_full.children())[2])[:-1]
        # )

        # a = list(self.single_clf_full.children())
        # print(a, len(a))
        # a1 = list(a[0].children())
        # # print(a1, len(a1))
        # print(a1[:-4], len(a1[:-4]))
        # # sys.exit()

        # self.single_clf_core = nn.Sequential(
        #     # *list(list(self.single_clf_full.children())[0])[:-4]
        #     *list(a1[:-4])
        # )

        # self.feature_extractor = nn.Sequential(*list(patch_clf_model.children())[:-extract_layers])

        # take only the weights of 'patch classifier' part, disregard original Top layer(s)
        if exp_type == 'PBC':
            self.single_clf_core = self.single_clf_full.feature_extractor       # PBC & old
        elif exp_type == 'DC':
            self.single_clf_core = feat_ext         # DC only
        else:
            self.single_clf_core = self.single_clf_full.feature_extractor   

        # cont = 0
        # for name, param in self.single_clf_core.named_parameters():
        #     print(cont, name)#, param)
        #     cont+=1
        # import sys
        # sys.exit()

        # a linha acima le o modelo, transforma em lista e pega o [2]
        #  que eh o proprio modelo (0 e 1 sao a declaracao do __init__ do
        #  SingleBreast..) e nele remove a ultima layer que sao os dois
        #  res blocks e a FC final

        # self.model_dict = {view: self.single_clf_core for view in SIDES.LIST}


    def forward(self, x):
        # x input comes in dict of views also

        # x[SIDES.CC].shape: torch.Size([1, 3, 1152, 896])
        # x[SIDES.MLO].shape: torch.Size([1, 3, 1152, 896])

        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        hidden2 = torch.cat([x1_2, x2_2], dim=1)

        # hidden = {
        #     view: self.model_dict[view](x[view]) for view in SIDES.LIST
        # }

        # hidden[SIDES.CC].shape: torch.Size([1, 2048, 36, 28])
        # hidden[SIDES.MLO].shape): torch.Size([1, 2048, 36, 28]))

        return hidden2

    # def single_forward(self, single_x, view):
    #     return self.model_dict[view](single_x)




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


        # cont = 0
        # # for name, param in self.two_views_clf.named_parameters():
        # for name, param in self.named_parameters():
        #     print(cont, name)#, param)
        #     cont+=1
        # import sys
        # sys.exit()


        # 03-Ago-2024
        # Now let's put the top layers above the two legs.
        #  It is based in MBConv blocks based in EfficientNet instances
        #  created here. The width of MBconv should be always fixed
        #  but when is EficientNets we are basing in their size.
        #  For ConvNext we create based in EficientNet-B0 sizes. 
        #  Maybe it should be always like that. Change and text afer 2024-thesis.


        print('Creating Side Mid Networking using:', network, ' and Top Block type: ', b_type, ' Qty: ', n_blocks)

        if network == 'Resnet50':
            input_channels = 4096       # From concatenation
            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output  
           
            if n_blocks == 1:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
            elif n_blocks == 2:
                self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)

        # elif network == 'EfficientNet-b4':
        elif 'EfficientNet' in network or 'convnext' in network:
            input_channels = 3584       # From concatenation+
            output_filters=2048
            

            if b_type == 'resnet':
                self.w_h = 9*7  # width and height of last layer output
                if n_blocks == 1:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                elif n_blocks == 2:
                    self.block1 = Bottleneck(inplanes=input_channels, planes=512, stride=2)
                    self.block2 = Bottleneck(inplanes=output_channels, planes=512, stride=2)
                else:
                    output_channels = 3584  # only FC

            elif b_type == 'mbconv':

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
                # Parameters for the entire model (stem, all blocks, and head)
                GlobalParams = collections.namedtuple('GlobalParams', [
                    'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
                    'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
                    'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])

                new_block = BlockArgs(num_repeat=1, kernel_size=3, stride=[strides], expand_ratio=2,
                                    input_filters=inplanes, output_filters=output_channels, se_ratio=0.25, id_skip=True) # para 1 block
                                    # input_filters=inplanes, output_filters=1792, se_ratio=0.25, id_skip=True) # para FC-only, 2-blocks
                                    # input_filters=inplanes, output_filters=1280, se_ratio=0.25, id_skip=True) # PARA EFICIENT-B0

                # Below line is based in patch_clf_model._global_params
                #print(feat_ext._global_params) --> to check originals for current EficientNet-Bx
                # orig dropout_rate=0.4
                # drop_connect_rate=0.2
                # new_global_params = GlobalParams(width_coefficient=1.4, depth_coefficient=1.8,
                #                                 image_size=380, dropout_rate=0.4, num_classes=5,
                #                                 batch_norm_momentum=0.99, batch_norm_epsilon=0.001,
                #                                 drop_connect_rate=0.2, depth_divisor=8, min_depth=None,
                #                                 include_top=True)

                # print('Changing parameters of Top Block(s)')
                # print('new_global_params: ', new_global_params)

                # below line for same params for blocks as main net
                block_args, global_params, image_size = new_block, feat_ext._global_params, [15, 15]
                # below line for custom params
                # block_args, global_params, image_size = new_block, new_global_params, [15, 15]

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
            #self.fc = nn.Linear(3584, 2)

        else:
            # AVGPOOL
            self.fc_pre = nn.Linear(2048* self.w_h, 1024)  #para aproveitar features espaciais Resnet 9*7 / EficientNet 36*28
            self.fc = nn.Linear(1024, 2)     # INCLUINDO - 2020-10-14 - para aproveitar features espaciais



    def forward(self, x):
        # x.shape: torch.Size([1, 3, 1152, 896])
        # print(x.shape)
        x = self.two_views_clf(x)       # out: torch.Size([1, 4096, 36, 28])
        # print(x.shape)
        if self.n_blocks == 1:
            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
        elif self.n_blocks == 2:

            x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
            x = self.block2(x)          # out: Resnet([1, 2048, 9, 7]) / Eficientnet [2, 2048, 36, 28] 

            # print(x.shape)

        if self.avg_pool:
            x = self.avgpool(x)             # out: torch.Size([1, 2048, 1, 1])
            x = torch.flatten(x, 1)         # out: torch.Size([1, 2048])
        else:
            # NO AVGPOOL
            x = x.view(-1, 2048* self.w_h)  # para aproveitar features espaciais  Resnet 9* 7 / Efiencet 36*28
            x = self.fc_pre(x)


        # print(x.shape)

        x = self.fc(x)                  # out: torch.Size([1, 2]

        return x






# ARRUMAR AQUI PARA CV
class TwoViewsTOPBreastClassifier(nn.Module):

    def __init__(self, device, model_file, network):
        super(TwoViewsTOPBreastClassifier, self).__init__()

        self.single_clf_full = SingleBreastClassifier(device, model_file, network)
        
        self.single_clf_full.load_state_dict(torch.load(model_file, map_location=device))

        # # Remove FC's

        # if network == 'Resnet50':
        #     # retrive layers from full clf, except last (avgpool, fc)
        #     #self.single_clf_core = self.single_clf_full.patch_clf_core
        #     full = list(self.single_clf_full.children())
        #     patch_clf = list(full[0].children())
        #     top_layer = list(full[1].children())
        #     self.single_clf_core = nn.Sequential(
        #         *patch_clf,
        #         *top_layer[:-1]        # termina em  AdaptiveAvgPool2d(output_size=(1, 1))] incluido
        #     )

        # elif network == 'EfficientNet-b4':

        # take the patch_clf part
        self.single_clf_core = self.single_clf_full.feature_extractor

        # then the top layer without the last FC and flatten (includes AVGPool)
        full = list(self.single_clf_full.children())
        patch_clf = list(full[0].children())
        top_layer = list(full[1].children())
        self.top_layer_block = nn.Sequential(*top_layer[:-1])


        # a linha acima le o modelo, transforma em lista e pega o [2]
        #  que eh o proprio modelo (0 e 1 sao a declaracao do __init__ do
        #  SingleBreast..) e nele remove a ultima layer que sao os dois
        #  res blocks e a FC final

        # self.model_dict = {view: self.single_clf_core for view in SIDES.LIST}


    def forward(self, x):
        # x input comes in dict of views also

        # x[SIDES.CC].shape: torch.Size([1, 3, 1152, 896])
        # x[SIDES.MLO].shape: torch.Size([1, 3, 1152, 896])


        # x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        # x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        # hidden = torch.cat([x1_2, x2_2], dim=1)

        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        x1_2 = self.top_layer_block(x1_2)
        x2_2 = self.top_layer_block(x2_2)
        hidden = torch.cat([x1_2, x2_2], dim=1)

        # hidden = {
        #     view: self.model_dict[view](x[view]) for view in SIDES.LIST
        # }

        # hidden[SIDES.CC].shape: torch.Size([1, 2048, 1, 1])
        # hidden[SIDES.MLO].shape): torch.Size([1, 2048, 1, 1])

        return hidden



class SideTOPBreastModel(nn.Module):

    connections = 256
    output_size = (4, 2)

    def __init__(self, device, model_file, network):
        super(SideTOPBreastModel, self).__init__()
        self.two_views_clf = TwoViewsTOPBreastClassifier(device, model_file, network)
        self.fc = nn.Linear(2048*2, 2)

    def forward(self, x):

        x = self.two_views_clf(x)

        #x_side = torch.cat([x[SIDES.CC], x[SIDES.MLO]], dim=1)
        # x_side.shape: torch.Size([1, 4096, 1, 1])
        # x = torch.flatten(x_side, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x.shape: torch.Size([1, 2])

        return x


# IMPLEMENTA APENAS FC depois dos blocos. Com e sem AVGPool.
# Ok para eficient net
class SideFCBreastModel(nn.Module):
    def __init__(self, device, model_file, network, avgpool=False):
        super(SideFCBreastModel, self).__init__()
        self.avgpool_flag = avgpool
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network)
        if self.avgpool_flag:
            if 'b0' in network:
                inplanes = 1280*2     # tamanho dos mapas concatenados do MIDBreast avg 2D
            elif 'b4' in network:
                inplanes = 1280*2 
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            if 'b0' in network:
                inplanes = 1280*2*36*28     # tamanho dos mapas concatenados do MIDBreast
            elif 'b4' in network:
                inplanes = 1280*2*36*28
        self.fc = nn.Linear(inplanes, 2)
    def forward(self, x):
        x = self.two_views_clf(x)
        if self.avgpool_flag:
            x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# IMPLEMENTA FUNCAO {media, OU, etc} ENTRE AS DUAS PERNAS CC e MLO
class SideFunctionBreastModel(nn.Module):
    def __init__(self, device, model_file, network):
        super(SideFunctionBreastModel, self).__init__()
        self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network)
        # self.fc = nn.Linear(2048*2, 2)
    def forward(self, x):
        print(x.shape)
        x = self.two_views_clf(x)
        print(x.shape, x)  # torch.Size([1, 4]) tensor([[-1.0210,  1.0837, -0.4706,  0.4218]], device='cuda:0')
        print(torch.log_softmax(x, 1))
        x_cc = x[:, 1]      # take both malignant predictions
        x_mlo = x[:, 3]
        print(x_mlo.shape, x_mlo, x_cc)
        a = torch.tensor([[x_cc, x_mlo]]).cuda(0)
        x = torch.mean(a)
        x = torch.tensor([[0, x]]).cuda(0)

        
        print(x.shape, x)
        print('FIX ME - precisa ajustar os valores das predicao antes da mean....')
        sys.exit(0)
        return x


class SideMIDThinBreastModel(nn.Module):
    """ Usa cada parte patch clf com treino de patch clf """

    connections = 256
    output_size = (4, 2)

    def __init__(self, device):
        super(SideMIDThinBreastModel, self).__init__()

        # instanciate resnet with 5 outputs
        self.patch_clf_ = MyResnet50(num_classes=5)

        # load weights from patch classifier
        self.patch_clf_.load_state_dict(torch.load(MODEL_PATCH_CLF, map_location=device))

        # Remove FC's
        self.patch_clf = nn.Sequential(
            *list(self.patch_clf_.children())[:-2],     # remove FC layer
        )

        # Modulos superiores
        self.block1 = Bottleneck(inplanes=4096, planes=512, stride=2)
        self.block2 = Bottleneck(inplanes=2048, planes=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # parece que Sinai eh 7x7
        self.fc = nn.Linear(2048, 2)


    def forward(self, x):
        # x.shape: torch.Size([1, 3, 1152, 896])

        # x = self.two_views_clf(x)

        x1_2 = self.patch_clf(x[:, 0:3, :, :])
        x2_2 = self.patch_clf(x[:, 3:6, :, :])
        hidden = torch.cat([x1_2, x2_2], dim=1)

        # x[SIDES.CC].shape: torch.Size([1, 2048, 36, 28])
        #x_side = torch.cat([x[SIDES.CC], x[SIDES.MLO]], dim=1)
        # x_side.shape: torch.Size([1, 4096, 36, 28])

        x = self.block1(hidden)     # x.shape: torch.Size([1, 2048, 18, 14])
        x = self.block2(x)          #  x.shape: torch.Size([1, 2048, 9, 7])
        x = self.avgpool(x)         # x.shape: torch.Size([1, 2048, 1, 1]) - AVGPOOL tirar?
        x = torch.flatten(x, 1)     # x.shape: torch.Size([1, 2048])
        x = self.fc(x)              # x.shape: torch.Size([1, 2])

        return x


class TwoViewsTOPBreastClassifier_OLD(nn.Module):   # ANTES EFiciente net/CV

    def __init__(self, device):
        super(TwoViewsTOPBreastClassifier, self).__init__()

        self.single_clf_full = SingleBreastClassifier(device, MODEL_FULL_CLF)

        # retrive layers from full clf, except last (avgpool, fc)
        #self.single_clf_core = self.single_clf_full.patch_clf_core
        level1 = list(self.single_clf_full.children())
        level2 = list(level1[2].children())
        level3 = list(level2[8].children())
        self.single_clf_core = nn.Sequential(
            *level2[:-1],
            *level3[:-1]        # termina em  AdaptiveAvgPool2d(output_size=(1, 1))] incluido
        )

        # a linha acima le o modelo, transforma em lista e pega o [2]
        #  que eh o proprio modelo (0 e 1 sao a declaracao do __init__ do
        #  SingleBreast..) e nele remove a ultima layer que sao os dois
        #  res blocks e a FC final

        self.model_dict = {view: self.single_clf_core for view in SIDES.LIST}


    def forward(self, x):
        # x input comes in dict of views also

        # x[SIDES.CC].shape: torch.Size([1, 3, 1152, 896])
        # x[SIDES.MLO].shape: torch.Size([1, 3, 1152, 896])

        x1_2 = self.single_clf_core(x[:, 0:3, :, :])
        x2_2 = self.single_clf_core(x[:, 3:6, :, :])
        hidden = torch.cat([x1_2, x2_2], dim=1)

        # hidden = {
        #     view: self.model_dict[view](x[view]) for view in SIDES.LIST
        # }

        # hidden[SIDES.CC].shape: torch.Size([1, 2048, 1, 1])
        # hidden[SIDES.MLO].shape): torch.Size([1, 2048, 1, 1])

        return hidden




# integramos NOAVG_Pool na principal com parametros - PODE APAGAR
# class SideMIDBreastModel_NoAvgPool(nn.Module):
#     """ Usa cada parte patch clf com treino de full image classifier
#         (pegando apenas a parte patch classifier)
#         REMOVE AVGPOOL para manter geometria espacial
#         Troca
#     """

#     connections = 256
#     output_size = (4, 2)

#     def __init__(self, device, model_file, network,):
#         super(SideMIDBreastModel_NoAvgPool, self).__init__()

#         self.two_views_clf = TwoViewsMIDBreastClassifier(device, model_file, network)

#         self.block1 = Bottleneck(inplanes=4096, planes=512, stride=2)
#         self.block2 = Bottleneck(inplanes=2048, planes=512, stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))     # REMOVENDO - 2020-10-14 - para aproveitar features espaciais

#         self.fc_pre = nn.Linear(2048* 9* 7, 256)        # INCLUINDO - 2020-10-14 - para aproveitar features espaciais

#         # self.fc = nn.Linear(2048, 2)  # REMOVENDO - 2020-10-14 - para aproveitar features espaciais
#         self.fc = nn.Linear(256, 2)     # INCLUINDO - 2020-10-14 - para aproveitar features espaciais


#     def forward(self, x):
#         # x.shape: torch.Size([1, 3, 1152, 896])

#         x = self.two_views_clf(x)   # out: torch.Size([1, 4096, 36, 28])
#         x = self.block1(x)          # out: torch.Size([1, 2048, 18, 14])
#         x = self.block2(x)          # out: torch.Size([1, 2048, 9, 7])

#         x = x.view(-1, 2048* 9* 7)  # INCLUINDO - 2020-10-14 - para aproveitar features espaciais
#         x = self.fc_pre(x)          # INCLUINDO - 2020-10-14 - para aproveitar features espaciais

#         # x.shape: torch.Size([1, 2048])
#         x = self.fc(x)              # out: torch.Size([1, 2]


#         return x
