#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from first_model import data_prep_christoph as data_prep


#%% packages
import torch
import seaborn as sns
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


#%% Hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = .01
HIDDEN_SIZE = 4
INPUT_SIZE = data_prep.X_Scaled.shape[1]
OUTPUT_SIZE = data_prep.y.shape[1]

#%% train test split
X_train_val, X_test, y_train_val, y_test = train_test_split(data_prep.X_Scaled, data_prep.y, test_size=0.1, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

#%% dataset
class AnxietyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X) # alternativ: self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = AnxietyDataset(X=X_train, y=y_train)
val_dataset = AnxietyDataset(X=X_val, y=y_val)
test_dataset = AnxietyDataset(X=X_test, y=y_test)

#%% dataloader
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=True)

#%% model class
class DeepRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepRegression, self).__init__()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_hidden = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_hidden(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x

model = DeepRegression(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)

# %% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

#%% training loop
loss_list_train, loss_list_val = [], []
for epoch in range(EPOCHS):
    loss_epoch_train, loss_epoch_val = 0, 0

    # model training
    for i, (X_batch, y_true_batch) in enumerate(train_dataloader):
        # forward pass --> output: y_pred
        y_pred_batch = model(X_batch)
        
        # calculate losses
        loss = loss_fn(y_pred_batch, y_true_batch)

        # backward pass (calc grads)
        loss.backward()

        # update model parameter
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        # (opt.) store losses in list
        loss_epoch_train += loss.item() / len(train_dataloader)
    loss_list_train.append(loss_epoch_train)

    # model validation
    for i, (X_batch_val, y_true_batch_val) in enumerate(val_dataloader):
        # 1. create predictions
        with torch.no_grad():
            y_pred_batch_val = model(X_batch_val)

        # 2. extract losses
        loss_val = loss_fn(y_pred_batch_val, y_true_batch_val)
        loss_epoch_val += loss_val.item() / len(val_dataloader)
    loss_list_val.append(loss_epoch_val)


    # (opt.) print loss to console
    print(f"Epoch {epoch}, Train-Loss: {loss_epoch_train:.3f}, Val-Loss: {loss_epoch_val:.3f}")

# %% elegantere Lösung
sns.lineplot(data=[loss_list_train, loss_list_val])

# %% create test predictions
with torch.no_grad():
    for (X_test, y_true_test) in test_dataloader:
        y_pred_test = model(X_test).numpy()


# %% calculate R^2 for test data
r2_score(y_pred=y_pred_test, y_true=y_true_test.numpy())

# %% parameter studie

