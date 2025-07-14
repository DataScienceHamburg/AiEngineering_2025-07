#%% packages
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
os.getcwd()
from PIL import Image
import seaborn as sns
import time

#%% HYPERPARAMETER
BATCH_SIZE = 128
EPOCHS = 20
LR = 0.001

#%% transform, load data
transform_steps = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

# %% visualize images
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


sample_img = Image.open("data/train/Positive/00001.jpg")
print(f"Image dimensions: {sample_img.size}")
sample_img_transformed = transform_steps(sample_img)



#%% Dataset and Dataloader
train_ds =torchvision.datasets.ImageFolder(root="data/train", transform=transform_steps)
test_ds =torchvision.datasets.ImageFolder(root="data/test", transform=transform_steps)

trainloader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

#%% get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)
imshow(torchvision.utils.make_grid(images, nrow=2))
# %% Neural Network setup
class ImageClassificationNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)  # (BS, 6, 30, 30)
        x = self.relu(x)  # 
        x = self.pool(x)  # (BS, 6, 15, 15)
        x = self.conv2(x)  # (BS, 16, 13, 13)
        x = self.relu(x) 
        x = self.pool(x)  # (BS, 16, 6, 6)
        x = torch.flatten(x, 1) # (BS, 16*6*6)
        x = self.fc1(x)  # (BS)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
        
        

#%% init model
model = ImageClassificationNet()      

#%%
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# %% training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
losses = []
start_time = time.time()
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, data in enumerate(trainloader, 0):
        X_batch, y_batch = data
        # Move data to the same device as model
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        # zero gradients
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(X_batch)
        
        # calc losses
        loss = loss_fn(y_pred, y_batch.reshape(-1, 1).float())
        
        # backward pass
        loss.backward()

        # update weights
        optimizer.step()
        
        # losses
        loss_epoch += loss.item()
    losses.append(loss_epoch)
    print(f"Epoch: {epoch}: {loss_epoch}")
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds")

#%% loss history
sns.lineplot(x=range(EPOCHS), y = losses)


# %% test
y_test = []
y_test_pred = []
for i, data in enumerate(testloader, 0):
    inputs, y_test_temp = data
    with torch.no_grad():
        y_test_hat_temp = model(inputs).round()
    
    y_test.extend(y_test_temp.numpy())
    y_test_pred.extend(y_test_hat_temp.numpy())

#%% Confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=train_ds.classes, yticklabels=train_ds.classes)
# %%
acc = accuracy_score(y_test, y_test_pred)
print(f'Accuracy: {acc*100:.2f} %')
# %% Naive classifier

#%% assignment of folder names
# Print mapping between folder names and labels
print("\nFolder to label mapping:")
print(f"Folder '{train_ds.classes[0]}' -> Label 0")
print(f"Folder '{train_ds.classes[1]}' -> Label 1")

# %% model parameters count
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# %% How much memory of CUDA is used?
import torch.cuda
print(torch.cuda.memory_summary(device=None, abbreviated=False))

# %%
torch.cuda.is_available()


#%% 