import os
from glob import glob
import random

import numpy as np
import scipy
import pandas as pd

import matplotlib.pyplot as plt

import torch.utils.data as data

import torch
import torchvision
from torchvision import models, transforms

import shutil

from PIL import Image

#os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# image paths
img_paths = glob('./data/train/images/*.npy')

#print(len(img_paths))
#img_path_0 = img_paths[0]
#print(img_path_0)

dst_folder = './data_nii/Test_Img'

# label paths
label_paths = glob('./data/train/labels/*.npy')

#print(len(label_paths))
#label_path_0 = label_paths[11]
#print(label_path_0)

# df 구성
df_paths = './data_nii/TrainValid_ground_truth.csv'
df = pd.read_csv(df_paths)

#print(df.head())

"""
img_data = np.load(img_paths[0])
print(img_data.shape)
print(img_data.dtype)

img_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
print(img_name)
"""

# transformed은 ToTensor만 일단 하기
# 나중에 crop해주기, 윗부분 날리는 것과 아예 RLQ빼고 다 날리는 것도 생각하기

train_transform = transforms.Compose([
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset
# dataframe을 읽어와서 필요한 정보 저정하기
class APPEDataset(data.Dataset):
    def __init__(self, img_paths, split, label_paths, df_paths, transform=None):

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

        # df 불러오기
        df = pd.read_csv(self.df_paths)

        # 전체 슬라이스 개수 계산
        self.slice_info = [] # (CT_idx, slice_idx) 저장
        for CT_idx, img_path in enumerate(self.img_paths):
            # num_slice를 slice_per_CT에 넣기
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            #img_name = os.path.splitext(os.path.basename(img_name_))[0]
            num_slices = df[df['id'].str.contains(img_name)].shape[0] - 1 # df에서 Image_name을 만족하는 열을 가져와서 -1하면 그 CT의 slice정보를 알수있어

            for slice_idx in range(num_slices):
                self.slice_info.append((CT_idx, slice_idx))

    def __len__(self):
        return len(self.slice_info)  # 전체 슬라이스 개수 반환

    def __getitem__(self, index):

        # index, CT_idx, slice_idx
        CT_idx, slice_idx = self.slice_info[index]

        # CT 불러오기, 이미 환자 한명의 CT를 가져옴 --> shape (512,512,94)
        #nii_data = nib.load(self.img_paths[CT_idx])
        #nii_img = nii_data.get_fdata()

        img_data = np.load(self.img_paths[CT_idx])

        #img_name_ = os.path.splitext(os.path.basename(self.img_paths[CT_idx]))[0]
        img_name = os.path.splitext(os.path.basename(self.img_paths[CT_idx]))[0]
        #print(img_name)

        # 파일 이름 가져와서 이름과 일치하는 label가져오기
        label_path = [item for item in label_paths if img_name in item][0]
        #nii_label_data = nib.load(nii_label_path)
        #nii_label = nii_label_data.get_fdata()
        label_data = np.load(label_path)

        # 이미 로드된 CT에서 해당 슬라이스만 추출
        #img_slice = nii_img[:512, :512, slice_idx]  # (512, 512) 가끔씩 513 이런게 있네
        #label_slice = nii_label[:512, :512, slice_idx]  # (512, 512)

        img_slice = img_data[:512,:512,slice_idx]
        label_slice = label_data[:512,:512,slice_idx]

        img_slice = np.clip(img_slice, 0, 1000)
        min_val = np.min(img_slice)
        max_val = np.max(img_slice)
        img_slice = (255 * ((img_slice - min_val) / (max_val - min_val))).astype(np.uint8)

        # CNN 입력을 위해 [C, H, W] 형태로 변환
        # 이건 이전에 내가 한 코드 보고 바꿀지 안바꿀지 결정하기 --> 이전에도 channel추가했었네 (3, 256, 256)
        img_slice = np.expand_dims(img_slice, axis=-1)  # (512, 512, 1)
        label_slice = np.expand_dims(label_slice, axis=-1)  # (512, 512, 1)

        # Transform 적용
        if self.transform:
            img_slice = self.transform(img_slice)
            label_slice = self.transform(label_slice)

        #print(img_slice.shape, img_slice.min(), img_slice.max())
        #print(label_slice.shape, label_slice.min(), label_slice.max())

        #assert 0

        return img_slice, label_slice
    
# 일단 train/ valiation 나누지 말고 만들어 보기
def dataloader(batch_size):
    train_dataset = APPEDataset(img_paths, 'train', label_paths, df_paths, transform = train_transform)

    val_dataset = APPEDataset(img_paths, 'val', label_paths, df_paths, transform = train_transform)

    # train_dataset 데이터 로더 작성
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    valiation_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    return train_dataloader, valiation_dataloader

"""    
train_dataset = APPEDataset(img_paths, label_paths, df_paths, transform = train_transform)

# 동작 확인

index = 0
print(train_dataset.__getitem__(index)[0].shape)
print(train_dataset.__getitem__(index)[1].shape)
print(train_dataset.__getitem__(index)[1])
"""

"""
import os
import numpy as np
import nibabel as nib
from glob import glob


# 입력 및 출력 폴더 설정
input_folder = './data_nii/TrainValid_Image/*/'
output_folder = './data/train/images/'

# 출력 폴더 생성 (존재하지 않으면)
os.makedirs(output_folder, exist_ok=True)

# .nii.gz 파일 목록 가져오기
img_paths = glob(os.path.join(input_folder, "*.nii.gz"))

for img_path in img_paths:
    # 파일 로드
    img = nib.load(img_path)
    img_data = img.get_fdata()
    
    # 파일 이름 추출 (확장자 제거)
    filename = os.path.basename(img_path).replace('.nii.gz', '.npy')
    
    # numpy 배열로 저장
    np.save(os.path.join(output_folder, filename), img_data)
    
    print(f"Saved: {filename}")"

# 입력 및 출력 폴더 설정
input_folder = './data_nii/TrainValid_Mask/2_Train,Valid_Mask/'
output_folder = './data/train/labels/'

# 출력 폴더 생성 (존재하지 않으면)
os.makedirs(output_folder, exist_ok=True)

# .nii.gz 파일 목록 가져오기
img_paths = glob(os.path.join(input_folder, "*.nii.gz"))

for img_path in img_paths:
    # 파일 로드
    img = nib.load(img_path)
    img_data = img.get_fdata()
    
    # 파일 이름 추출 (확장자 제거)
    filename = os.path.basename(img_path).replace('.nii.gz', '.npy')
    
    # numpy 배열로 저장
    np.save(os.path.join(output_folder, filename), img_data)
    
    print(f"Saved: {filename}")"


import os
import shutil

# 폴더 경로 설정
test_images_dir = "./data/test/images/"
train_labels_dir = "./data/train/labels/"
test_labels_dir = "./data/test/labels/"

# 테스트 라벨 폴더가 없으면 생성
os.makedirs(test_labels_dir, exist_ok=True)

# 테스트 이미지 파일 이름 목록 (확장자 제거)
test_image_names = [os.path.splitext(f)[0] for f in os.listdir(test_images_dir) if os.path.isfile(os.path.join(test_images_dir, f))]

# Train labels에서 해당하는 파일 찾기
moved_count = 0
for label_file in os.listdir(train_labels_dir):
    label_path = os.path.join(train_labels_dir, label_file)

    # Train labels의 파일 이름이 test image의 파일 이름을 포함하고 있는 경우 이동
    if any(image_name in label_file for image_name in test_image_names):
        shutil.move(label_path, os.path.join(test_labels_dir, label_file))
        moved_count += 1

print(f"총 {moved_count}개의 라벨 파일이 이동되었습니다.")"
"""