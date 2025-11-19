#%%

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from first_model import data_prep

#%% packages
import torch
import seaborn as sns
from sklearn.metrics import r2_score

#%% convert numpy arrays to tensors
X_tensor = torch.from_numpy(data_prep.X_scaled)
y_tensor = torch.from_numpy(data_prep.y)

#%%
y_tensor.shape

#%% model class
# class LinearRegression(torch.nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearRegression, self).__init__()
#         self.linear = torch.nn.Linear(in_features=input_size, out_features=output_size)

#     def forward(self, x):
#         x = self.linear(x)
#         return x

#%% Hyperparameters
EPOCHS = 100
LR = .1
HIDDEN_SIZE = 4
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



model = DeepRegression(input_size=X_tensor.shape[1], output_size=y_tensor.shape[1], hidden_size=HIDDEN_SIZE)

# %% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.MSELoss()

#%% training loop
loss_list = []
for epoch in range(EPOCHS):
    # forward pass --> output: y_pred
    y_pred = model(X_tensor)

    # calculate losses
    loss = loss_fn(y_pred, y_tensor)

    # backward pass (calc grads)
    loss.backward()

    # update model parameter
    optimizer.step()

    # reset gradients
    optimizer.zero_grad()

    # (opt.) store losses in list
    loss_list.append(loss.item())

    # (opt.) print loss to console
    print(f"Epoch {epoch}, Loss: {loss.item():.3f}")
# %%
# sns.lineplot(x=list(range(EPOCHS)), y=loss_list)
# %% elegantere Lösung
sns.lineplot(data=loss_list)
# %%
