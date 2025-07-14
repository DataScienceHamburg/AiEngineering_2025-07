#%% packages
import torch
import pandas as pd
import numpy as np
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
#%% data prep
cars_file_path = "data/car_price_dataset.csv"
cars = pd.read_csv(cars_file_path)
cars.head()

#%%
cars.columns

#%%
# cars.describe()
print(cars.info())

#%%

#%% Exploratory Data Analysis
# how many car brands are there?
cars['Brand'].nunique()

#%%
cars['Brand'].value_counts()

#%% Define independent and dependent variables
cars_one_hot = pd.concat([cars, pd.get_dummies(cars['Transmission'], prefix='Transmission')],axis=1)
# get all columns that start with 'Brand_'
transmission_cols = [col for col in cars_one_hot.columns if col.startswith('Transmission_')]

X_filtered = cars_one_hot[['Mileage', 'Engine_Size', 'Year'] + transmission_cols]
y = cars['Price']

#%% one hot encode the brand

#%% separate data into train and test
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y, test_size=0.2, random_state=42)


#%% normalize the data with StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% visualize the data including the regression line
sns.regplot(data=cars, x='Mileage', y='Price', line_kws={'color': 'red'})


#%% convert to torch tensors
X_train_torch = torch.from_numpy(X_train_scaled).float()
X_test_torch = torch.from_numpy(X_test_scaled).float()
y_train_torch = torch.from_numpy(y_train.values).float()
y_test_torch = torch.from_numpy(y_test.values).float()

#%%
class DeepRegTorch(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=30):
        super(DeepRegTorch, self).__init__()
        self.relu = nn.ReLU()
        self.input = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
# %%
input_dim = X_train_torch.shape[1]
output_dim = 1
model = DeepRegTorch(input_size=input_dim, output_size=output_dim, hidden_size=5)


# %% Loss Function
loss_fn = nn.MSELoss()

#
# %% Hyperparameter
LR = 0.01
EPOCHS = 300
BATCH_SIZE = 128

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)



# %% train the model
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for batch in range(0, X_train_torch.shape[0], BATCH_SIZE):
        X_batch = X_train_torch[batch:batch+BATCH_SIZE]
        y_batch = y_train_torch[batch:batch+BATCH_SIZE]
        # set gradients to zero
        optimizer.zero_grad()
        
        # forward pass
        y_pred = model(X_batch)
        
        # loss calc
        loss = loss_fn(y_pred, y_batch.reshape(-1, 1))
        
        # backward pass
        loss.backward()
        
        # update parameters
        optimizer.step()
                    
        loss_epoch += float(loss.data)
        
    # store losses
    losses.append(loss_epoch/X_train_torch.shape[0])
    
    # print loss
    if (epoch % 10 == 0):
        print(f"Epoch {epoch}: Loss {loss.data}")
        
#%% visualize the loss
sns.lineplot(x=range(EPOCHS), y = losses)

#%% test the model
with torch.no_grad():
    y_test_pred = model(X_test_torch)



#%%
sns.regplot(x=y_test, y=y_test_pred.detach().numpy().flatten(), line_kws={'color': 'red'})

#%% calculate the R-squared score
r2_score(y_test, y_test_pred.detach().numpy().flatten())

#%% Hypothesis to test
# H0: Scaling does not improve the performance of the model
# ['Mileage']  -> 0.302
# ['Mileage', 'Engine_Size']  -> 0.414
# ['Mileage', 'Engine_Size', 'Owner_Count']  -> 0.414
# ['Mileage', 'Engine_Size', 'Owner_Count', 'Doors']  -> 0.414
# ['Mileage', 'Engine_Size', 'Brand_?']  -> 0.413
