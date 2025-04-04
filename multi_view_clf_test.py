# 2 views CLASSIFIER Improved - test script
#
# Test inference for 2 views mammograms with 2024 nwtworks
#
# run: python3 2views_clf_test.py -c [cc image file] -m [mlo image file]
#
# DGPP 12/Dec/2024


import argparse
import numpy as np
import torch
from torch.autograd import Variable
import cv2

from two_views_net_2024 import SideMIDBreastModel, SideTOPBreastModel, SideFCBreastModel, SideFunctionBreastModel


# ****************************************** New

DATASET =  'CBIS-DDSM'          #   'CBIS-DDSM' or 'VINDR_MAMMO' 
NETWORK =  'EfficientNet-b3'    #   'EfficientNet-b3' or 'convnext_base'
EXPERIMENT_TYPE = 'PBC'         #   'DC', 'PBC'

# ******************************************


TOPOLOGY = 'side_mid_clf'     # Only
DEVICE = 'gpu'
gpu_number = 0


# Configs for single model assembling
TOP_MODE = 'TOP_EF_NET' 
TOP_LAYER_N_BLOCKS = 2
STRIDES = 2
TOP_LAYER_BLOCK_TYPE = 'mbconv'
USE_AVG_POOL = True
TRAIN_DS_MEAN = 13369   # Mean of all files in training 


class LoadModel():
    def __init__(self, device, network, topology=None) -> None:
        self.device = device
        self.network = network
        self.topology = topology


    def get_single_model_file(self):
        """
        Load singke view classifier first, from file, in network
        """

        if self.network == 'EfficientNet-b3':
            if EXPERIMENT_TYPE == 'PBC':
                model_file = 'models/2024-05-13-21h48m_50ep_best_model_AUC_08393_Best_CBIS_EFB3-CVUBP.pt'
            elif EXPERIMENT_TYPE == 'DC':
                model_file = 'models/2024-05-14-18h55m_50ep_best_model_AUC_08548_Best_CBIS_EFB3_CVUD.pt'
        elif self.network == 'convnext_base':
            model_file = 'models/2024-08-06-18h01m_50ep_best_model_AUC_08187_CVUBP_VINDR_CONVNEXT.pt'

        print('Model Single:', model_file)
        return model_file

    def load_2views_model(self):
        """
        Load complete model from file in network
        """

        if self.network == 'EfficientNet-b3':
            if EXPERIMENT_TYPE == 'PBC':
                self.model_file = 'models/2024-10-01-14h36m_50ep_best_model_AUC_08643_Best_2-Views-CBIS-EFB3-CVUBP.pt'
            elif EXPERIMENT_TYPE == 'DC':
                self.model_file = 'models/2024-07-01-12h34m_50ep_best_model_AUC_08856_BEST_2024_Ef-B3_CBIS.pt'
        elif self.network == 'convnext_base':
            self.model_file = 'models/2024-08-07-18h35m_50ep_best_model_AUC_08655_ConvNext_VinDr_Collab.pt'

        print('Model 2views: ', self.model_file)

        # Then load full model
        self.model.load_state_dict(torch.load(self.model_file, map_location=self.device))

        return self.model


    def load_model(self):

        model_file = self.get_single_model_file()

        if self.topology == 'side_top_clf':
            model = SideTOPBreastModel(self.device, model_file, self.network)
        elif self.topology == 'side_mid_clf':
            model = SideMIDBreastModel(self.device, model_file, self.network, TOP_LAYER_N_BLOCKS,
                                    b_type=TOP_LAYER_BLOCK_TYPE, avg_pool=USE_AVG_POOL,
                                    strides=STRIDES,
                                    exp_type = EXPERIMENT_TYPE,
                                    dataset = DATASET)

        elif self.topology == 'side_fc_clf':
            model = SideFCBreastModel(self.device, model_file, self.network, avgpool=True)
        elif self.topology == 'side_function_clf':
            model = SideFunctionBreastModel(self.device, model_file, self.network)
        else:
            raise NotImplementedError(f"Net type error: {self.topology}")

        self.model = model.to(self.device)

        return self.model, self.device



# normalize accordingly for model
def standard_normalize(image):
    image = np.float32(image)
    if 'CBIS-DDSM' in DATASET:
        image -= TRAIN_DS_MEAN
        image /= 65535
    elif 'VINDR_MAMMO' in DATASET:
        image /= 65535
        image -= np.mean(image)
        image /= np.maximum(np.std(image), 10**(-5))

    return image    


def make_prediction(image_cc, image_mlo, model, device):
    """ Execute deep learning inference
        Evaluates 2-views model
        inputs: [vector of] image
                norm = standard normalization function (from dataloader)
        output: full image mask 
        """
    img_cc = standard_normalize(image_cc)
    img_mlo = standard_normalize(image_mlo)

    img_cc_t = torch.from_numpy(img_cc.transpose(2, 0, 1))
    img_mlo_t = torch.from_numpy(img_mlo.transpose(2, 0, 1))
    batch_t = torch.cat([img_cc_t, img_mlo_t], dim=0)
    batch_t = batch_t.unsqueeze(0)

    # prediction
    with torch.no_grad():
        model.eval()
        input = Variable(batch_t.to(device))
        output_t = model(input)

    pred = output_t.squeeze()
    pred = torch.softmax(pred, dim=0)

    return pred, batch_t


def simple_prediction(image_cc, image_mlo, model, device):
    tta_predictions = np.array([])

    for i in range(1,2):
        aug_image_cc = image_cc
        aug_image_mlo = image_mlo
        prediction, tensor = make_prediction(aug_image_cc, aug_image_mlo, model, device)
        tta_predictions = np.append(tta_predictions, prediction[1].cpu().detach().numpy())
    
    return tta_predictions


# <<<<<<<<<<<<<<<<<< main <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
def main():


    ap = argparse.ArgumentParser(description='[Poli-USP] Two Views Breast Cancer inference 2024 Version')
    ap.add_argument("-c", "--cc", required=True, help="CC image file.")
    ap.add_argument("-m", "--mlo", required=True, help="MLO image file.")
    # ap.add_argument("-d", "--model", help="two-views detector model")
    # ap.add_argument("-a", "--aug", help="select to use translation augmentation: -a true")

    args = vars(ap.parse_args())

    file_cc = args['cc']
    file_mlo = args['mlo']

    # Assign DEVICE
    if (DEVICE == "gpu") and torch.backends.cudnn.is_available():
        device = torch.device("cuda:{}".format(gpu_number))
    else:
        device = torch.device("cpu")

    print(f'Evaluating Two Views Input with dataset {DATASET} with network {NETWORK}', end='')
    print(f' with 2-views model file')

    MyModel = LoadModel(device, NETWORK, TOPOLOGY)

    # First assemble 2-views network with the single view models
    model, device = MyModel.load_model()

    # Then load the trained 2-views model
    model = MyModel.load_2views_model()


    file_cc = '/home/dpetrini/devel/DNN/pytorch/full_clf_2views/samples/Calc-Test_P_00041_LEFT_CC.png'
    file_mlo = '/home/dpetrini/devel/DNN/pytorch/full_clf_2views/samples/Calc-Test_P_00041_LEFT_MLO.png'

    image = cv2.imread(file_cc, cv2.IMREAD_UNCHANGED)

    image_cc = np.zeros((*image.shape[0:2], 3), dtype=np.uint16)
    image_cc[:, :, 0] = image
    image_cc[:, :, 1] = image
    image_cc[:, :, 2] = image


    image = cv2.imread(file_mlo, cv2.IMREAD_UNCHANGED)

    image_mlo = np.zeros((*image.shape[0:2], 3), dtype=np.uint16)
    image_mlo[:, :, 0] = image
    image_mlo[:, :, 1] = image
    image_mlo[:, :, 2] = image

    tta_predictions = simple_prediction(image_cc, image_mlo, model, device)
    pred = np.mean(tta_predictions)

    print(f'\nPrediction: {pred:.4f}')


if __name__ == '__main__':
    main()
