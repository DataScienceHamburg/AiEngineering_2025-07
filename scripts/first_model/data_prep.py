#%% packages
import os
import kagglehub
import pandas as pd
import numpy as np
import torch

#%% Data import
path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")

print("Path to dataset files:", path)

anxiety_file = os.path.join(path, "enhanced_anxiety_dataset.csv")
df_anxiety = pd.read_csv(anxiety_file)
df_anxiety.head(2)

#%% Explorative Daten Analyse (EDA)
df_anxiety.columns

#%%
df_anxiety.shape

#%% Behandlung von kategorischen Daten
df_anxiety_dummies = pd.get_dummies(df_anxiety, dtype=int, drop_first=True)
df_anxiety_dummies.head(2)
df_anxiety_dummies.shape

#%% Korrelationsmatrix
numerical_features = df_anxiety.select_dtypes(include=['int64', 'float64'])
corr = numerical_features.corr()

#%%
import seaborn as sns
sns.heatmap(corr, annot=True, cmap="coolwarm")

#%% zusammenhang sleep hours and anxiety level
sns.regplot(data=df_anxiety_dummies, x="Sleep Hours", y= "Anxiety Level (1-10)", color="blue", line_kws={"color": "red"})

#%% trenne unabhängige / abhängige Variable
y = np.array(df_anxiety_dummies[["Anxiety Level (1-10)"]], dtype=np.float32)
y.shape

#%%
X = np.array(df_anxiety_dummies.drop(columns=["Anxiety Level (1-10)"]), dtype=np.float32)
X.shape

#%% Data scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
