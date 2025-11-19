#%%
import os
import kagglehub
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import torch

# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)

#%%
path_train = os.path.join(path, "train")
path_test = os.path.join(path, "test")

#%% data preprocessing
preprocess_steps = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()
])
# %% Hyperparameters
BATCH_SIZE = 256
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% Dataset
trainval_dataset = torchvision.datasets.ImageFolder(root=path_train, transform=preprocess_steps)
test_dataset = torchvision.datasets.ImageFolder(root=path_test, transform=preprocess_steps)

validation_ratio = 0.2
train_size = int((1-validation_ratio) * len(trainval_dataset))
val_size = len(trainval_dataset) - train_size
train_dataset, val_dataset = random_split(dataset=trainval_dataset, lengths=[train_size, val_size])
print(f"Train Size: {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")


# %% DataLoader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

#%% Modellklasse und -instanz
class ImageClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels= 1, out_channels=6, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(576, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)  # out: (BS, 6, 30, 30)
        x = self.relu(x)
        x = self.pool(x)  # out: (BS, 6, 15, 15)
        x = self.conv2(x) # out: (BS, 16, 13, 13)
        x = self.relu(x)
        x = self.pool(x)  # out: (BS, 16, 6, 6)
        x = self.flatten(x)  # out: (BS, 16*6*6)
        x = self.fc1(x)  # out: (BS, 64)
        x = self.fc2(x)  # out: (BS, 1)
        return x

model = ImageClassificationModel().to(DEVICE)
# dummy_input = torch.randn(BATCH_SIZE, 1, 32, 32)
# model(dummy_input).shape
# %% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()

# %%
train_losses, val_losses = [], []
for epoch in range(EPOCHS):
    # Training
    train_loss_epoch = 0
    for i, (X_train, y_train) in enumerate(train_dataloader):
        # copy data to DEVICE
        X_train = X_train.to(DEVICE)
        y_train = y_train.to(DEVICE)
        
        # forward pass
        y_train_pred = model(X_train)
        # loss calc
        loss = loss_fn(y_train_pred, y_train.reshape(-1, 1).float())
        # backward pass
        loss.backward()

        # weight update
        optimizer.step()

        # zero gradients
        optimizer.zero_grad()

        # extract losses
        train_loss_epoch += loss.item() / len(train_dataloader)
    train_losses.append(train_loss_epoch)

    # Validation
    val_loss_epoch = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_dataloader:
            # copy to DEVICE
            X_val_batch = X_val_batch.to(DEVICE)
            y_val_batch = y_val_batch.to(DEVICE)
            
            y_val_pred = model(X_val_batch)
            val_loss = loss_fn(y_val_pred, y_val_batch.reshape(-1, 1).float())
            val_loss_epoch += val_loss.item() / len(val_dataloader)
    val_losses.append(val_loss_epoch)
    print(f"Epoch {epoch}: Trainloss {train_loss_epoch}, Val-loss {val_loss_epoch}")



# %% store model
torch.save(obj=model.state_dict(), f="binary_classification.pt")


