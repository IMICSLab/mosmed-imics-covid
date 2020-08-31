#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 23:00:04 2020

@author: Ernest Namdar
"""

import re
import pandas as pd
import os
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import copy
import SimpleITK as sitk
import pydicom
import scipy
from scipy import ndimage as nd
import json
import cv2
from iMICSdset_utils.coco import COCO
# import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
from scipy.ndimage import zoom


def read_nfti_img(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data

def read_nrrd_img(path):
    data, header = nrrd.read(path)
    return data, header

def read_img_mask_pair(img_path, msk_path):
    img = read_nfti_img(img_path)
    msk, header = read_nrrd_img(msk_path)
    #print(dir(header))
    return img, msk


class iMICSDataset(Dataset):
    """iMICS dataset."""
    def __init__(self, pr_root, target_image_size, cohort="all", shuffle=False):
        self.pr_dir = pr_root
        self.cohort = cohort
        self.shuffle = shuffle
        self.target_image_size = target_image_size
        directories = os.listdir(pr_root)
        mask_dirs = []
        img_dirs = []
        if self.cohort == "all":
            for folder in directories:
                if "_Masks" in folder:
                    mask_dirs.append(folder)
                elif "CT" in folder:
                    img_dirs.append(folder)
        else:
            for folder in directories:
                if cohort in folder:
                    if "_Masks" in folder:
                        mask_dirs.append(folder)
                    else:
                        img_dirs.append(folder)
        mask_dirs = sorted(mask_dirs)
        img_dirs = sorted(img_dirs)
        print(mask_dirs)
        print(img_dirs)
        if len(mask_dirs) != len(img_dirs):
            print("Error! Some folders were not identified correctly!")

        img_holder = []
        msk_holder = []
        pat_id_holder = []
        lbl_holder = []
        img_size_holder = []
        msk_size_holder = []
        count = 0
        for i in range(len(img_dirs)):
            group = int(re.findall('\d+', img_dirs[i])[0])
            img_directory = os.path.join(self.pr_dir, img_dirs[i])
            # print(img_directory)
            msk_directory = os.path.join(self.pr_dir, mask_dirs[i])
            img_list = os.listdir(img_directory)
            img_list = sorted(img_list)
            msk_list = os.listdir(msk_directory)
            msk_list = sorted(msk_list)
            if len(img_list) != len(msk_list):
                print("Error! Images and masks are not matched!")

            for j in range(len(img_list)):
                count += 1
                if count % 50 == 0:
                    print("Processing Patient# ", count)
                img_path = os.path.join(img_directory, img_list[j])
                msk_path = os.path.join(msk_directory, msk_list[j])
                img, msk = read_img_mask_pair(img_path, msk_path)
                if img.shape != msk.shape:
                    print("Error! Image and Mask do not have the same size!", img_list[j].strip(".nii.gz"))
                    print("ImgSize:", img.shape, "MskSize:", msk.shape)
                    msk, _ = read_nrrd_img(os.path.join(self.pr_dir, "Resized_Mks", msk_list[j]))
                    if img.shape != msk.shape:
                        print("Still Image and Mask do not have the same size!")
                    else:
                        print("The problem is solved")
                    # continue
                img_size_holder.append(img.shape)
                msk_size_holder.append(msk.shape)
                if img.shape != self.target_image_size:
                    #print("image size and target size before resize:", img.shape, self.target_image_size)
                    sc1 = self.target_image_size[0]/img.shape[0]
                    sc2 = self.target_image_size[1]/img.shape[1]
                    sc3 = self.target_image_size[2]/img.shape[2]
                    sc = (sc1, sc2, sc3)
                    tmp = zoom(img, sc)
                    img = tmp
                    tmp2 = zoom(msk, sc)
                    msk = tmp2
                    #print("image size and target size after resize:", img.shape, self.target_image_size)
                img = img.reshape(1, img.shape[0],img.shape[1],img.shape[2])
                msk = msk.reshape(1, msk.shape[0],msk.shape[1], msk.shape[2])
                img_holder.append(img)
                msk_holder.append(msk)
                lbl_holder.append(group)
                pat_id_holder.append(img_list[j].strip(".nii.gz"))
        self.IMG = np.concatenate(img_holder, axis=0)
        self.MSK = np.concatenate(msk_holder, axis=0)
        self.ImgSize = img_size_holder
        self.MskSize = msk_size_holder
        self.PatID = pat_id_holder
        self.Lbl = lbl_holder

        if self.shuffle is True:
            p = np.random.permutation(self.IMG.shape[0])
            self.IMG = self.IMG[p]
            self.MSK = self.MSK[p]
            self.Lbl = self.Lbl[p]
            self.ImgSize = self.ImgSize[p]
            self.MskSize = self.MskSize[p]
            self.PatID = self.PatID[p]

    def __len__(self):
        return (self.IMG.shape[0])

    def __getitem__(self, idx):
        img = self.IMG[idx, :, :, :]
        msk = self.MSK[idx, :, :, :]
        lbl = self.Lbl[idx]
        pid = self.PatID[idx]
        sample = {'image': img, 'mask': msk, 'label': lbl, 'PatID':pid}
        return sample


if __name__ == "__main__":
    pr_dir = "../studies"
    cohort = "CT-4" # or CT-1, or CT-2, or CT-3, or CT-4 or all
    # using max(set(dset.ImgSize), key=dset.ImgSize.count) = 512,512,45
    target_image_size = (256,256,32)
    dset = iMICSDataset(pr_dir, target_image_size=target_image_size, cohort='all')

    # plt.figure()
    # plt.imshow(dset.IMG[200])
    # plt.figure()
    # plt.imshow(dset.MSK[200])