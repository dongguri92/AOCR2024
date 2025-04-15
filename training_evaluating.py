import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models
import os
import segmentation_models_pytorch as smp

import matplotlib.pyplot as plt

from torch.amp import GradScaler, autocast

from PIL import Image

from segmentation_models_pytorch.losses import DiceLoss

def dice_score(preds, targets, threshold=0.5, eps=1e-6):
    preds = torch.sigmoid(preds)  # 로짓 -> 확률로 변환
    preds = (preds > threshold).float()  # Threshold 적용하여 Binary Map 생성

    intersection = (preds * targets).sum(dim=(1,2,3))  # 배치별 Intersection
    union = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))  # 배치별 Union

    dice = (2. * intersection + eps) / (union + eps)  # Dice Coefficient
    return dice.mean()  # 배치 평균 Dice Score 반환

# BCE + Dice loss 정의
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(mode='binary')

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class SupervisedLearning():

    def __init__(self, train_loader, validation_loader, model_name):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.model_name = model_name
        self.model = models.modeltype(self.model_name)
        self.model = self.model.to(self.device)

        print("Completed loading your network.")

        #self.criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

        self.criterion = nn.BCEWithLogitsLoss()  # 시그모이드 적용 전 로짓 사용

        #self.criterion = smp.utils.losses.DiceLoss()

        # 그래프를 위한 리스트 초기화
        self.train_loss_history = []
        self.train_dice_history = []
        self.valid_loss_history = []
        self.valid_dice_history = []

    def train(self, epochs, lr, l2): # epoch, lr, l2는 argparse에서 가져오기

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)
        scaler = GradScaler("cuda")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            total_dice_score = 0.0

            for batch_idx, (images, masks) in enumerate(self.train_loader):
                images, masks = images.float().to(self.device), masks.float().to(self.device)

                optimizer.zero_grad()

                with autocast("cuda"):  # Mixed Precision 적용
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                scaler.scale(loss).backward()  # Loss Scaling 적용
                scaler.step(optimizer)  # Optimizer step 수행
                scaler.update()  # GradScaler 업데이트

                # Dice Score 계산
                dice = dice_score(outputs, masks)

                #outputs = self.model(images)
                #outputs = torch.sigmoid(outputs_)

                #loss = self.criterion(outputs, masks)
                #loss.backward()
                #optimizer.step()

                total_loss += loss.item()
                #total_dice_score += (1 - loss.item())
                total_dice_score += dice.item()

                # Print loss every 4 batches
                if (batch_idx + 1) % 4 == 0:
                    avg_loss = total_loss / (batch_idx + 1)
                    avg_dice = total_dice_score / (batch_idx + 1)
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], "
                          f"Loss: {avg_loss:.4f}, Dice Score: {avg_dice:.4f}")

            avg_train_loss = total_loss / len(self.train_loader)
            avg_train_dice = total_dice_score / len(self.train_loader)
            self.train_loss_history.append(avg_train_loss)
            self.train_dice_history.append(avg_train_dice)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Dice Score: {avg_train_dice:.4f}")


            # validation평가
            with torch.no_grad():
                self.model.eval()
                valid_loss = 0
                valid_dice_score = 0

                for images, masks in self.validation_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)

                    valid_loss += loss.item()
                    #valid_dice_score += (1 - loss.item())
                    valid_dice_score += dice.item()

                avg_valid_loss = valid_loss / len(self.validation_loader)
                avg_valid_dice = valid_dice_score / len(self.validation_loader)
                self.valid_loss_history.append(avg_valid_loss)
                self.valid_dice_history.append(avg_valid_dice)
                print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_valid_loss:.4f}, Validation Dice Score: {avg_valid_dice:.4f}")

        # Training 완료 후 그래프 출력
        self.plot_metrics(save_path="./save/training_results.png")

        # 모델 저장
        torch.save(self.model.state_dict(), "./save/unet_0403.pth")
        print("saved.....")

    def plot_metrics(self, save_path="training_metrics.png"):
        """Loss & Dice Score를 그래프로 시각화하고 저장하는 함수"""
        epochs = range(1, len(self.train_loss_history) + 1)

        plt.figure(figsize=(12, 5))

        # Loss 그래프
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_loss_history, 'b-o', label="Train Loss")
        plt.plot(epochs, self.valid_loss_history, 'r-o', label="Valid Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        # Dice Score 그래프
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_dice_history, 'b-o', label="Train Dice Score")
        plt.plot(epochs, self.valid_dice_history, 'r-o', label="Valid Dice Score")
        plt.xlabel("Epochs")
        plt.ylabel("Dice Score")
        plt.title("Dice Score Curve")
        plt.legend()

        # 그래프 저장
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 그래프 저장 완료: {save_path}")

        plt.show()