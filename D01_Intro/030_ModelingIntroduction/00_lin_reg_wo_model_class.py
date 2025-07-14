#%%
import numpy as np
import pandas as pd
import torch
import seaborn as sns
from sklearn.linear_model import LinearRegression

#%% data import
cars_file_path = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file_path)
cars.head()

#%% visualise data with regression plot
sns.regplot(data=cars, x= 'wt', y='mpg', line_kws={'color': 'red'})

#%% separate independent / dependent feature
X = cars[['wt']]
y = cars['mpg']
# %% calculate linear regression coefficient with sklearn including intercept
model = LinearRegression(fit_intercept=True)
model.fit(X, y)
print(f'slope: {model.coef_}')
print(f'intercept: {model.intercept_}')
# %% data prep: conversion to tensors
# alternative X_np = np.array(X)
X_tensor = torch.from_numpy(X.to_numpy(dtype=np.float32))
y_tensor = torch.from_numpy(y.to_numpy(dtype=np.float32)).reshape(-1, 1)
# y_tensor.shape
# %% training
w = torch.rand(1, requires_grad=True, dtype=torch.float32)
b = torch.rand(1, requires_grad=True, dtype=torch.float32)

EPOCHS = 1000
LR = 0.001
loss_total = []
for _ in range(EPOCHS):
    loss_epoch = 0
    for i in range(len(X)):
        # forward pass
        y_pred = X_tensor[i] * w + b
        
        # calculate loss
        loss = torch.pow(y_pred - y_tensor[i], 2)        
        
        # backward pass
        loss.backward()
        
        # update weights and biases
        with torch.no_grad():
            # w = w-w_grad*LR
            w -= w.grad * LR
            
            b -= b.grad * LR
            
        # reset gradients
        w.grad.zero_()
        b.grad.zero_()
        
        # update loss_epoch
        loss_epoch += loss
    print(loss_epoch)
    loss_total.append(loss_epoch.item())
            
            

# %% visualise losses
sns.lineplot(x = range(EPOCHS), y=loss_total)
loss_total
# %%
