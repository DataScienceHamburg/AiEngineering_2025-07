#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from first_model import data_prep

#%% packages
import torch
import seaborn as sns
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader

#%% Hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = .1
HIDDEN_SIZE = 4
INPUT_SIZE = data_prep.X_scaled.shape[1]
OUTPUT_SIZE = data_prep.y.shape[1]
#%% dataset
class AnxietyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X) # alternativ: self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = AnxietyDataset(X=data_prep.X_scaled, y=data_prep.y)

#%% dataloader
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=False)


#%% model class
class DeepRegression(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(DeepRegression, self).__init__()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=hidden_size)
        self.linear_out = torch.nn.Linear(in_features=hidden_size, out_features=output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x

model = DeepRegression(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)

# %% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

#%% training loop
loss_list = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, (X_batch, y_true_batch) in enumerate(dataloader):
        # forward pass --> output: y_pred
        y_pred_batch = model(X_batch)
        # print(f"y true batch: {y_true_batch}")
        
        # calculate losses
        loss = loss_fn(y_pred_batch, y_true_batch)

        # backward pass (calc grads)
        loss.backward()

        # update model parameter
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        # (opt.) store losses in list
        loss_epoch += loss.item() / len(dataloader)
    loss_list.append(loss_epoch)

    # (opt.) print loss to console
    print(f"Epoch {epoch}, Loss: {loss_epoch:.3f}")
# %%
# sns.lineplot(x=list(range(EPOCHS)), y=loss_list)
# %% elegantere Lösung
sns.lineplot(data=loss_list)
# %% save state dictionary
torch.save(obj=model.state_dict(), f="model001.pt")

