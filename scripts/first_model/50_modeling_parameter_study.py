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
from sklearn.model_selection import train_test_split


#%% Hyperparameters
EPOCHS = 100
BATCH_SIZE = 256
LR = .01
HIDDEN_SIZE = 4
INPUT_SIZE = data_prep.X_scaled.shape[1]
OUTPUT_SIZE = data_prep.y.shape[1]

#%% Parameter study
import pandas as pd
import itertools

parameter_study = {
    "epochs": [10, 20, 40],
    "batch_size": [32, 64, 128],
    "learning_rate": [0.01, 0.001]
}

#%% Create a dataframe with each possible combination
param_keys = list(parameter_study.keys())
param_values = list(parameter_study.values())
all_combinations = list(itertools.product(*param_values))
parameter_df = pd.DataFrame(all_combinations, columns=param_keys)
parameter_df



#%% train test split
X_train_val, X_test, y_train_val, y_test = train_test_split(data_prep.X_scaled, data_prep.y, test_size=0.1, random_state=42)

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

#%% parameter study - iterate over each parameter combination
r2_scores = []

for idx, params in parameter_df.iterrows():
    print(f"\n{'='*60}")
    print(f"Simulation {idx + 1}/{len(parameter_df)}")
    print(f"Parameters: epochs={params['epochs']}, batch_size={params['batch_size']}, learning_rate={params['learning_rate']}")
    print(f"{'='*60}")
    
    # Create dataloaders with current batch_size
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=int(params['batch_size']), shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=int(params['batch_size']), shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=False)
    
    # Initialize model
    model = DeepRegression(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    loss_fn = torch.nn.MSELoss()
    
    # Training loop
    for epoch in range(int(params['epochs'])):
        loss_epoch_train, loss_epoch_val = 0, 0
        
        # Model training
        model.train()
        for i, (X_batch, y_true_batch) in enumerate(train_dataloader):
            # Forward pass
            y_pred_batch = model(X_batch)
            
            # Calculate losses
            loss = loss_fn(y_pred_batch, y_true_batch)
            
            # Backward pass
            loss.backward()
            
            # Update model parameters
            optimizer.step()
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Store losses
            loss_epoch_train += loss.item() / len(train_dataloader)
        
        # Model validation
        model.eval()
        with torch.no_grad():
            for i, (X_batch_val, y_true_batch_val) in enumerate(val_dataloader):
                y_pred_batch_val = model(X_batch_val)
                loss_val = loss_fn(y_pred_batch_val, y_true_batch_val)
                loss_epoch_val += loss_val.item() / len(val_dataloader)
        
        # Print loss every 10 epochs or last epoch
        if (epoch + 1) % 10 == 0 or epoch == int(params['epochs']) - 1:
            print(f"Epoch {epoch + 1}/{int(params['epochs'])}, Train-Loss: {loss_epoch_train:.3f}, Val-Loss: {loss_epoch_val:.3f}")
    
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        for (X_test_batch, y_true_test) in test_dataloader:
            y_pred_test = model(X_test_batch).numpy()
    
    # Calculate R^2 for test data
    r2_test = r2_score(y_pred=y_pred_test, y_true=y_true_test.numpy())
    r2_scores.append(r2_test)
    print(f"Test R^2 Score: {r2_test:.4f}")

# Add R^2 scores to parameter_df
parameter_df['r2_score'] = r2_scores

#%% Display results
print("\n" + "="*60)
print("PARAMETER STUDY RESULTS")
print("="*60)
print(parameter_df)
print("\n")
print(f"Best R^2 Score: {parameter_df['r2_score'].max():.4f}")
print("\nBest parameters:")
best_params = parameter_df.loc[parameter_df['r2_score'].idxmax()]
print(best_params)


# %%
