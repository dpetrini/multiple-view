# Read images and augments - Pytorch Dataset Loader
# 
# Loader for 4 images full - place mammo images em python dicts
#
# Read from folder datapath where has CBIS images with categorias top {malign, normal}
# as final folder in datapth. Create exams list and divide intervals
# 
# Author: Daniel Petrini
#
# August/2020 ~2025

import math
import os
from random import randint
import random
import sys
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import cv2

from img_process import flip, affine
from constants import SIDES

#TRAIN_DS_MEAN = 78 if size_8 else 20143  # mean from full train images (w/ crop)
TRAIN_DS_MEAN = 13369 # todo dataset V2 (paper2021). Before:13655

# using according to paper, also to fit better in GPU
HEIGHT = 1152
WIDTH = 896
N_CHANNELS = 3

MAX_RAM_PAIRS = 10 #700 #140     # Qty of pairs of images to load in RAM (140 good for 8GB)

class MyDataset(Dataset):
    """ Custom dataloader for 2-views 
        Supports CBIS-DDSM and VinDr-Mammo, by reading info from file name
    """
    def __init__(self, image_path, set_type='train', dataset='CBIS-DDSM'):
        self.path = image_path
        self.train = False

        if set_type not in ['train', 'val', 'test']:
            print('Please input type of set [train, val, test] ', os.path.basename(__file__))
            sys.exit()

        self.set_type = set_type
        self.dataset = dataset
        self.categories = ['malign', 'benign']      # Vindr has no biopsy, but we use like this anyway
        self.exam_list = self.get_dataset(self.categories, dataset=dataset)

        self.selection = []

        self.count_malig = 0
        self.count_benign = 0

        # Augmentation
        self.limit = 0.2
        self.max_brightness = 65535
        self.angle = 25
        self.shear = 12
        self.translate = 0
        self.gaussian_blur = False
        self.zoom_in = 0.20
        self.zoom_out = 0.20

        self.debug = False


        # allocate buffers
        self.max_img_ram = MAX_RAM_PAIRS
        self.exam_array = np.empty((len(SIDES.LIST), HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint16)
        self.img_cc = np.empty((HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint16)
        self.img_mlo = np.empty((HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint16)
        self.img_cc_t = torch.empty([N_CHANNELS, WIDTH, HEIGHT], dtype=torch.float32)
        self.img_mlo_t = torch.empty([N_CHANNELS, WIDTH, HEIGHT], dtype=torch.float32)
        self.batch_t = torch.empty([N_CHANNELS*2, WIDTH, HEIGHT], dtype=torch.float32)

        # Not used, was for test set random

        # # Calculate and prepare sets
        # train_size = math.floor(len(self.exam_list)*0.76)
        # test_size = math.floor(len(self.exam_list)*0.10)
        # val_size = len(self.exam_list) - train_size - test_size

        # if self.set_type == 'train':
        #     self.exam_list = self.exam_list[0: train_size]
        #     self.train = True
        # elif self.set_type == 'val':
        #     self.exam_list = self.exam_list[train_size: train_size + val_size]
        # elif self.set_type == 'test':
        #     self.exam_list = self.exam_list[:test_size]

        # load new structure : selection (common for the 2views project)
        # fill a dictionary with separate same-side images and labels
        for exam in self.exam_list:
            self.selection.append({
                'CC': exam['CC'],
                'MLO': exam['MLO'],
                'side': exam['side'],
                'label': exam['label']
                })

        print(f'[DataLoader] Read images {self.set_type} size: {len(self.exam_list)} pairs {self.path}')
        #print(self.selection[-1])

        # OPTIMIZED
        # Load on RAM
        num_images = 0
        if self.set_type in ('train'):   # , 'val'
            print(f'[DataLoader] Loading {self.set_type} images on RAM...')
            num_images = len(self.selection)
            # Allocate buffer for ALL exams
            self.all_exam_array = np.empty((self.max_img_ram,
                                            len(SIDES.LIST), HEIGHT, WIDTH, N_CHANNELS),
                                           dtype=np.uint16)
            for i, datum in enumerate(self.selection):
                if i >= self.max_img_ram:
                    print('Reached max ram size')
                    break
                # load a pair exam images in self.exam_array buffer then place in all_exam buf
                self.load_exam_array(datum)
                # print(i)
                self.all_exam_array[i] = self.exam_array
                num_images = i+1


            print(f'[DataLoader] Loaded {num_images} pairs of images (CC, MLO): {self.all_exam_array.nbytes} bytes.')



    # Perform data augmentation in training data
    def transform(self, image): #, target):

        if self.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)

        if self.debug:
            cv2.imshow('Img antes Aug', image)

        # intensity shift
        beta = self.limit * random.uniform(-1, 1)   # <0 demages with black pixels from image (full)
        image = cv2.add(image, beta*self.max_brightness)

        # rotate, translation, scale and shift augs
        angle = randint(-self.angle, self.angle)
        trans_x = randint(-self.translate, self.translate)
        trans_y = randint(-self.translate, self.translate)

        if randint(0, 1) == 0:
            scale = 1 + random.uniform(0, self.zoom_out)  # decrease zoom # 0.1
        else:
            scale = 1 - random.uniform(0, self.zoom_in)   # increase zoom # 0.35

        shear = randint(-self.shear, self.shear)
        # AFFINE - all at once
        image = affine(image, angle, (trans_x, trans_y), scale, shear,
                       mode=cv2.BORDER_REFLECT)  # ######## Inserted 2020-0605

        # flip {0: vertical, 1: horizontal, 2: both, 3: none}
        flip_num = randint(0, 3)
        image = flip(image, flip_num)

        if self.debug:
            print(image.shape, image.dtype, np.mean(image), scale)
            cv2.imshow('Img Pos Aug', image)
            cv2.waitKey(0)

        image = self.standard_normalize(image)

        return image

    def passthrough(self, image):

        if self.gaussian_blur:
            image = cv2.GaussianBlur(image, (3, 3), 0.5)

        if self.debug:
            print(image.shape, image.dtype, np.mean(image))
            cv2.imshow('Img Val', image)
            cv2.waitKey(0)

        image = self.standard_normalize(image)

        return image


    def __len__(self):
        return len(self.exam_list)

    def show_image(self, image):
        #temp = np.uint16(image)
        temp = np.uint8(image)
        # temp =  cv2.normalize(temp, None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        temp = cv2.resize(temp, (temp.shape[1]//6, temp.shape[0]//6))
        cv2.imshow('name', temp)
        cv2.waitKey(0)
        return

    # # normalize accordingly for model
    # def standard_normalize(self, image):
    #     image = np.float32(image)
    #     image -= TRAIN_DS_MEAN      # Sinai
    #     image /= 65535          # float [-1,1] 65535 - just faster

    #     return image


    # normalize accordingly for model
    def standard_normalize(self, image):
        image = np.float32(image)
        # if self.dataset == 'CBIS-DDSM':
        if 'CBIS-DDSM' in self.dataset:
            # image -= np.mean(image)           # ---> NYU Style
            image -= TRAIN_DS_MEAN              # Sinai Style
            image /= 65535
        # elif self.dataset == 'VINDR_MAMMO':
        elif 'VINDR_MAMMO' in self.dataset:
            image /= 65535                        
            image -= np.mean(image)               
            image /= np.maximum(np.std(image), 10**(-5))

        return image

    # OPTIMIZED
    def load_exam_array(self, datum):
        """ Read image from disk. Load object self.exam_array fixed buffer """
        image_extension = ".png"
        for key in datum.keys():
            if key in  SIDES.LIST:
                img_dir = os.path.join(self.path, datum['label'])
                image_gray = cv2.imread(
                    # os.path.join(self.image_path, datum[key] + image_extension),
                    os.path.join(img_dir, datum[key]),
                    #os.path.join(self.image_path, datum[key]),
                    cv2.IMREAD_UNCHANGED)

                # if not USE_RESIZED_IMGS:
                #     # same conversion as in patch creation
                #     image_gray = cv2.resize(image, (WIDTH, HEIGHT),
                #                             interpolation=cv2.INTER_AREA)
                image = np.zeros((*image_gray.shape[0:2], 3), dtype=np.uint16)
                image[:, :, 0] = image_gray
                image[:, :, 1] = image_gray
                image[:, :, 2] = image_gray

                if key == 'CC':
                    self.exam_array[0] = image.reshape(1, *image.shape)
                elif key == 'MLO':
                    self.exam_array[1] = image.reshape(1, *image.shape)

        #return exam_array


    #OPTIMIZED with global allocation
    def __getitem__(self, idx):

        datum = self.selection[idx]  # for the labels

        # load 2 images from exam
        if self.set_type in ('train') and idx < self.max_img_ram:  # , 'val' 
            # print('loaded from RAM ', idx)
            self.exam_array = self.all_exam_array[idx]  # RAM Only
        else:
            # self.loaded_image_array = self.load_exam_array(datum, self.exam_array)    # DISK
            self.load_exam_array(datum)    # load pair of images from DISK to RAM buffer

        # AUGMENTATION (and changing to 32 bits float)
        if self.train:
            self.img_cc = self.transform(self.exam_array[0])
            self.img_mlo = self.transform(self.exam_array[1])
        else:
            self.img_cc = self.passthrough(self.exam_array[0])
            self.img_mlo = self.passthrough(self.exam_array[1])

        self.img_cc_t = torch.from_numpy(self.img_cc.transpose(2, 0, 1))
        self.img_mlo_t = torch.from_numpy(self.img_mlo.transpose(2, 0, 1))
        self.batch_t = torch.cat([self.img_cc_t, self.img_mlo_t], dim=0)

        # labels = torch.tensor((datum['label']), dtype=torch.long)
        labels = torch.tensor((1 if datum['label'] == 'malign' else 0), dtype=torch.long)

        return self.batch_t, labels



    def get_dataset(self, categories, dataset='CBIS-DDSM'):
        """ Read all CBIS-DDSM (or VinDr-Mammo - 2024) and place in dictionary (of file names)"""
        exam_list = []
        temp_list = [[], []]   # keep separated categories exam

        for k, category in enumerate(categories):

            file_list = [i for i in os.listdir(os.path.join(self.path, category)) if i.endswith('.png')]
            file_list.sort()
            total_files = len(file_list)
            count = 0

            if dataset == 'CBIS-DDSM':
                # Example file name: Calc-Training_P_02226_LEFT_MLO.png
                for j in range(total_files):

                    if file_list[j][:7] == 'Calc-Tr' or file_list[j][:7] == 'Mass-Tr':
                        sub = 16
                    elif file_list[j][:7] == 'Calc-Te' or file_list[j][:7] == 'Mass-Te':
                        sub = 12

                    if (j+1) != total_files:
                        if file_list[j][:sub+5] == file_list[j+1][:sub+5]:
                            # check if have same SIDE
                            if file_list[j][sub+5: sub+5+4] != file_list[j+1][sub+5: sub+5+4]:
                                continue
                            case = file_list[j][sub: sub+5]
                            if file_list[j][sub+6: sub+7] == 'R':
                                side = 'right'
                            elif file_list[j][sub+6: sub+7] == 'L':
                                side = 'left'
                            else:
                                print('Wrong image file name from dataset ', os.path.basename(__file__))
                                sys.exit()
                            cc_file = file_list[j]
                            mlo_file = file_list[j+1]

                            temp_list[k].append({
                                'case': case,
                                'CC': cc_file,
                                'MLO': mlo_file,
                                'side': side,
                                'label': category
                            })

                            count += 1

            elif dataset == 'VINDR_MAMMO':
                # Example file name: ff14000d3_R_MLO_B1_DC.png
                for j in range(total_files):

                    sub = 0

                    if (j+1) != total_files:
                        if file_list[j][:9] == file_list[j+1][:9]:
                            # check if have same SIDE
                            if file_list[j][10] != file_list[j+1][10]:
                                continue
                            case = file_list[j][:9]
                            if file_list[j][10] == 'R':
                                side = 'right'
                            elif file_list[j][10] == 'L':
                                side = 'left'
                            else:
                                print('Wrong image file name from dataset ', os.path.basename(__file__))
                                sys.exit()
                            cc_file = file_list[j]
                            mlo_file = file_list[j+1]

                            temp_list[k].append({
                                'case': case,
                                'CC': cc_file,
                                'MLO': mlo_file,
                                'side': side,
                                'label': category
                            })

                            count += 1


        #print('Occurrences: ', len(self.temp_list[0]), len(self.temp_list[1]))

        # compare size of 2 lists
        size1, size2 = len(temp_list[0]), len(temp_list[1])
        min_size = min(size1, size2)
        max_size = max(size1, size2)
        reminder = max_size - min_size

        # shuffle categories one each for final list
        for i in range(min_size):
            exam_list.append(temp_list[0][i])
            exam_list.append(temp_list[1][i])

        # if not same sizes, append higher in the end or ????????
        if reminder != 0:
            print('Accounting for list size differences')
            if size1 > size2:
                for i in range(min_size, size1):
                    exam_list.append(temp_list[0][i])
            elif size2 > size1:
                for i in range(min_size, size2):
                    exam_list.append(temp_list[1][i])

        malign_count = benign_count = 0
        
        for sample in exam_list:
            if sample['label'] == 'malign':
                malign_count += 1
            elif sample['label'] == 'benign':
                benign_count += 1
                

        print(f'Total: {len(exam_list)} pairs [CC, MLO], positive (malign): {malign_count}, negative (benign): {benign_count}')

        return exam_list
