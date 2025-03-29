import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import models
import segmentation_models_pytorch as smp

class SupervisedLearning():

    def __init__(self, train_loader, validation_loader, model_name):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.model_name = model_name
        self.model = models.modeltype(self.model_name)
        self.model = self.model.to(self.device)

        print("Completed loading your network.")

        self.criterion = nn.BCEWithLogitsLoss()
        #self.criterion = smp.utils.losses.DiceLoss()

    def train(self, epochs, lr, l2): # epoch, lr, l2는 argparse에서 가져오기

        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=l2)

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0  # epoch별 loss를 저장할 변수

            # Train loop
            for batch_idx, (data, target) in enumerate(self.train_loader):
                target = target.type(torch.LongTensor)
                data, target = data.float().to(self.device), target.float().to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                output = output.type(torch.float32)

                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()  # loss 합산

                # Print loss every 4 batches
                if (batch_idx + 1) % 4 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(self.train_loader)}], Loss: {total_loss / (batch_idx + 1):.4f}")

            # After the epoch, print the average train loss
            avg_train_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")

            # Validation evaluation
            with torch.no_grad():
                self.model.eval()
                valid_loss = 0

                for images, masks in self.validation_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    valid_loss += loss.item()

            avg_valid_loss = valid_loss / len(self.validation_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_valid_loss:.4f}")

        # 모델 저장
        torch.save(self.model.state_dict(), "./save/unet_0329.pth")
        print("saved.....")

"""        for epoch in range(epochs):
            total_loss = 0  # epoch별 loss를 저장할 변수

            for batch_idx, (data, target) in enumerate(self.train_loader):
                target = target.type(torch.LongTensor)
                data, target = data.float().to(self.device), target.float().to(self.device)

                optimizer.zero_grad()
                output = self.model(data)
                output = output.type(torch.float32)

                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() # loss 합산
                #print(total_loss)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(self.train_loader):.4f}")

        # validation 평가
        with torch.no_grad():
            self.model.eval()
            valid_loss = 0

            for images, masks in self.validation_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                valid_loss += loss.item()

        valid_loss /= len(self.validation_loader)
        print(f"EPOCH:{epoch + 1}, Loss:{valid_loss}")"""