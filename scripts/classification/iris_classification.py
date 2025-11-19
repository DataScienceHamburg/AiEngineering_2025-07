#%%
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, auc, roc_curve
import seaborn as sns

#%% Hyperparameter
LR = 0.01
EPOCHS = 200

#%% data loading
iris = load_iris()
X = iris.data
y = iris.target

#%% train-test split mit 20% Testanteil (Train, Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#%% erstelle IrisData als erbende Klasse von Dataset
class IrisData(Dataset):
    def __init__(self, X, y) -> None:
        super().__init__()
        self.X = torch.from_numpy(X.astype('float32'))
        self.y = torch.from_numpy(y.astype('int64'))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = IrisData(X=X_train, y=y_train)
test_data = IrisData(X=X_test, y=y_test)

#%% DataLoader
train_loader = DataLoader(dataset=train_data, batch_size=X.shape[0], shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=X.shape[0])

#%% Modellklasse
class MulticlassClassificationModel(torch.nn.Module):
    def __init__(self, num_features, num_targets, num_hidden):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=num_features, out_features=num_hidden)
        self.lin2 = torch.nn.Linear(in_features=num_hidden, out_features=num_targets)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        # x = self.softmax(x)  # wird nicht benötigt, weil es schon in der Verlustfkt. enthalten ist
        return x

class SimpleMulticlassClassificationModel(torch.nn.Module):
    def __init__(self, num_features, num_targets, num_hidden):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_features=num_features, out_features=num_targets)
        # self.softmax = torch.nn.LogSoftmax()

    def forward(self, x):
        x = self.lin1(x)
        return x

# %% Modellinstanz
num_targets = len(np.unique(y))
model = SimpleMulticlassClassificationModel(num_features=X.shape[1], num_targets=num_targets, num_hidden=6)
# %% Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.CrossEntropyLoss()

#%% Trainingsschleife
losses = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for X_train, y_train in train_loader:
        
        # forward pass
        y_pred_train = model(X_train)

        # calculate losses
        loss = loss_fn(y_pred_train, y_train)

        # calculate gradients
        loss.backward()

        # update parameters
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        # loss store
        loss_epoch += loss.item() / len(train_loader)
    losses.append(loss_epoch)
    print(f"Epoch {epoch}, Loss: {loss_epoch}")
# %%
sns.lineplot(losses)
# %% create preds for test data
with torch.no_grad():
    for (X_test, y_test) in test_loader:
        y_test_pred_logits = model(X_test)  # model predicts logits and not probabilities
        y_test_pred_probs = torch.softmax(input=y_test_pred_logits, dim= 1)  # logits -> softmax -> probabilities
        y_test_pred_class = torch.argmax(y_test_pred_probs, dim=1)  # probabilities of classes for each element --> predicted class

#%% confusion matrix
import matplotlib.pyplot as plt
cm = confusion_matrix(y_pred=y_test_pred_class, y_true=y_test)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.xlabel("Predicted label")
plt.ylabel("True label")

# %% Accuracy
accuracy_score(y_true=y_test, y_pred=y_test_pred_class)

#%% ROC curve
from sklearn.preprocessing import label_binarize

# Binarize the labels for multiclass ROC
y_test_binary = label_binarize(y_test, classes=[0, 1, 2])
y_test_pred_probs_np = y_test_pred_probs.numpy()

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 3

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_test_pred_probs_np[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve of class {iris.target_names[i]} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - Multiclass')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# %%
