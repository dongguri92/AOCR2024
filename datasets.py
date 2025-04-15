import os
from glob import glob

import pandas as pd

import numpy as np

import torch.utils.data as data

from PIL import Image

import nibabel as nib

import torch
import torchvision
from torchvision import models, transforms

# image path

img_paths = glob('./data/train/images/*.npy')
# (512,512,94), float64

# test label path
label_paths = glob('./data/train/labels/*.npy')
# (512,512,94), float64

# csv 파일 로드
df_paths = "/NAS-2/djk25/AOCR2024/TrainValid_ground_truth.csv"
df = pd.read_csv(df_paths)


# tranform은 우선 ToTensor만으로

train_transform = transforms.Compose([
    transforms.ToTensor(),
])
# transform에 이미지 변환 넣었는데 그러면 label에도 같이 적용이 되어서 일단은 그대로 두었음

validation_transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataframe을 읽어와서 필요한 정보 저정하기
class APPEDataset(data.Dataset):
    def __init__(self, split, img_paths, label_paths, df_paths, transform=None):

        self.img_paths = img_paths
        self.label_paths = label_paths
        self.df_paths = df_paths
        self.transform = transform
        self.split = split

        from sklearn.model_selection import train_test_split
        train_paths, val_paths = train_test_split(self.img_paths, train_size=0.9, random_state=1)

        if self.split == 'train':
            self.img_paths = train_paths
        elif self.split == 'val':
            self.img_paths = val_paths

        df = pd.read_csv(self.df_paths)

        # 전체 슬라이스 개수 계산
        self.slice_info = [] # (CT_idx, slice_idx) 저장
        for CT_idx, img_path in enumerate(self.img_paths):
            # num_slice를 slice_per_CT에 넣기
            # enumerate니까 CT별 index 정보도 포함됨
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            num_slices = df[df['id'].str.contains(img_name)].shape[0] - 1 # df에서 Image_name을 만족하는 열을 가져와서 -1하면 그 CT의 slice정보를 알수있어

            for slice_idx in range(num_slices):
                self.slice_info.append((CT_idx, slice_idx))

    def __len__(self):
        return len(self.slice_info)  # 전체 슬라이스 개수 반환

    def __getitem__(self, index):

        # index, CT_idx, slice_idx
        CT_idx, slice_idx = self.slice_info[index]

        # CT 불러오기, 이미 환자 한명의 CT를 가져옴 --> shape (512,512,94)
        img_data = np.load(self.img_paths[CT_idx])

        img_name = os.path.splitext(os.path.basename(self.img_paths[CT_idx]))[0]

        # 파일 이름 가져와서 이름과 일치하는 label가져오기
        label_path = [item for item in self.label_paths if img_name in item][0]
        label_data = np.load(label_path)

        # 이미 로드된 CT에서 해당 슬라이스만 추출
        img_slice = img_data[:512, :512, slice_idx]  # (512, 512) 가끔씩 513 이런게 있네
        label_slice = label_data[:512, :512, slice_idx]  # (512, 512)

        # 정규화
        img_slice = np.clip(img_slice, 0, 1000)
        min_val = np.min(img_slice)
        max_val = np.max(img_slice)
        img_slice = (255 * ((img_slice - min_val) / (max_val - min_val))).astype(np.uint8)

        img_slice = np.expand_dims(img_slice, axis=-1)  # (512, 512, 1)
        label_slice = np.expand_dims(label_slice, axis=-1)  # (512, 512, 1)

        # Transform 적용
        if self.transform:
            img_slice = self.transform(img_slice)
            label_slice = self.transform(label_slice)

        return img_slice, label_slice

def dataloader(batch_size):
    train_dataset = APPEDataset('train', img_paths, label_paths, df_paths, transform = train_transform)
    val_dataset = APPEDataset('train', img_paths, label_paths, df_paths, transform = validation_transform)

    # train_dataset 데이터 로더 작성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    # validation_dataset 데이터 로더 작성
    validation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    return train_dataloader, validation_dataloader