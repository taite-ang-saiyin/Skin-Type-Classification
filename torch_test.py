import numpy as np 
import pandas as pd 
from copy import deepcopy
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
label_index = {"dry": 0, "normal": 1, "oily": 2,"combination":3,"Damage":4,"Acne prone skin":5}
index_label = {0: "dry", 1: "normal", 2: "oily",3: "combination",4:"Damage",5:"Acne prone skin"}
def create_df(base):
    dd = {"images": [], "labels": []}
    for i in os.listdir(base):
        label = os.path.join(base, i)
        for j in os.listdir(label):
            img = os.path.join(label, j)
            dd["images"] += [img]
            dd["labels"] += [label_index[i]]
    return pd.DataFrame(dd)


EPOCHS = 20
LR = 0.1
STEP = 15
GAMMA = 0.1
BATCH_SIZE = 32
OUT_CLASSES = 6
IMG_SIZE = 224


class CloudDS(Dataset):
    def __init__(self, data, transform):
        super(CloudDS, self).__init__()
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, x):
        img, label = self.data.iloc[x, 0], self.data.iloc[x, 1]
        img = Image.open(img).convert("RGB")
        img = self.transform(np.array(img))
        
        return img, label
def get_dataloaders(batch_size=32, num_workers=0):
    train_df = create_df("config/train")
    val_df = create_df("config/valid")
    test_df = create_df("config/test")

    train_transform = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                transforms.RandomVerticalFlip(0.6),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

    transform = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor(),
                                   transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225])])

    train, testing = train_test_split(train_df, random_state=42, test_size=0.2)
    val, test = train_test_split(testing, random_state=42, test_size=0.5)

    train_ds = CloudDS(train, train_transform)
    val_ds = CloudDS(val, transform)
    test_ds = CloudDS(test, transform)  

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_dl, val_dl, test_dl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
def train():
    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, OUT_CLASSES)  # Update the final fully connected layer
    model = deepcopy(resnet).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)


    train_dl, val_dl, _ = get_dataloaders()

    best_model = deepcopy(model)
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0
        total = 0

        for data, target in train_dl:
            optimizer.zero_grad()
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_acc += (outputs.argmax(1) == target).sum().item()
            total += target.size(0)

        train_loss = running_loss / total
        train_acc = running_acc / total

        model.eval()
        val_loss, val_acc = 0.0, 0.0
        val_total = 0

        with torch.no_grad():
            for data, target in val_dl:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = criterion(outputs, target)

                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == target).sum().item()
                val_total += target.size(0)

        val_loss /= val_total
        val_acc /= val_total

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = deepcopy(model)

        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")


    torch.save(best_model.state_dict(), "best_resnet50_skin_type2.pth")

if __name__ == '__main__':
    train() 