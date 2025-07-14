#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import pandas as pd
# %% Hyperparameter
BATCH_SIZE = 2**6
LR = 0.001
EPOCHS = 100
 
# %% data import
studentsDataSet = pd.read_csv("data/Student_performance_data _.csv")
studentsDataSet.head()
# %% Select features and target
X = studentsDataSet[["Age","StudyTimeWeekly", "ParentalEducation", "Absences", "ParentalSupport", "Tutoring"]]
y = studentsDataSet[['GradeClass']]
 
# %% scale data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)
 
# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
# %% convert to float32
x_train_tensor = torch.from_numpy(np.array(X_train).astype(np.float32))
x_test_tensor = torch.from_numpy(np.array(X_test).astype(np.float32))
y_train_tensor = torch.from_numpy(np.array(y_train).astype(np.int64)).squeeze().long()
y_test_tensor = torch.from_numpy(np.array(y_test).astype(np.int64)).squeeze().long()
 
# %% dataset
class FlowerDataSet(Dataset):
    def __init__(self, x_in, y_in):
        self.x = x_in
        self.y = y_in
 
    def __len__(self):
        return self.x.shape[0]
   
    def __getitem__(self, index):
        return self.x[index], self.y[index]
 
# %% dataloader
train_dataloader = DataLoader(FlowerDataSet(x_train_tensor, y_train_tensor), batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(FlowerDataSet(x_test_tensor, y_test_tensor), batch_size=BATCH_SIZE, shuffle=True)
 
# %% check dims
#s.o.
 
# %% define class
class LinearRegressionFlower(nn.Module):
    def __init__(self, input_length, output_length, hidden_length):
        super(LinearRegressionFlower, self).__init__()
        self.input = nn.Linear(input_length, hidden_length)
        self.hidden = nn.Linear(hidden_length, hidden_length)
        self.output = nn.Linear(hidden_length, output_length)
        self.relu = nn.ReLU()
       
    def forward(self, x):
        x = self.input(x)
        x = self.relu(x)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x
 
# %% hyper parameters
# NUM_FEATURES = ...
# HIDDEN = ...
# NUM_CLASSES = ...
 
# %% create model instance
input_length = X.shape[1]
hidden_length = 5
output_length = y.nunique().item()
model = LinearRegressionFlower(input_length=input_length, output_length=output_length, hidden_length=hidden_length)
 
# %% loss function
loss_function = nn.CrossEntropyLoss()
 
# %% optimizer
optimizer = torch.optim.Adam(model.parameters(), LR)
 
# %% Training
epochloss = []
for epoIdx in range(EPOCHS):
    loss_epoch = 0
    for idx, (x_batch, y_batch) in enumerate(train_dataloader):
        optimizer.zero_grad()
        y_pred = model(x_batch)
        cur_loss = loss_function(y_pred, y_batch)
        loss_epoch += cur_loss.item()
       
        cur_loss.backward()
        optimizer.step()
    epochloss.append(cur_loss.item())
    print(f"Epoch {epoIdx} - Loss: {cur_loss.item()}")

# %% show losses over epochs
sns.lineplot(x=range(EPOCHS), y=epochloss, color='red')
 
# %% Confusion Matrix
from sklearn.metrics import confusion_matrix
# make predictions
y_pred = model(x_test_tensor)
y_pred = torch.argmax(y_pred, dim=1)

# calculate confusion matrix
cm = confusion_matrix(y_test_tensor, y_pred)

# plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# %% test the model
with torch.no_grad():
    correct = 0
    total = 0
    for x_batch, y_batch in test_dataloader:
        y_pred = model(x_batch)
        _, predicted = torch.max(y_pred, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    print(f"Accuracy: {correct/total}")
    
#%% Naive Classification with most frequent class
from sklearn.dummy import DummyClassifier

# create a dummy classifier that always predicts the most frequent class
dummy_classifier = DummyClassifier(strategy='most_frequent')

# fit the dummy classifier
dummy_classifier.fit(X_train, y_train)

# make predictions
y_pred = dummy_classifier.predict(X_test)

# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")






    







# %%
