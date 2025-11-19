#%%
import os
# import kagglehub
import pandas as pd
import numpy as np
import torch
# %%
# path = kagglehub.dataset_download("natezhang123/social-anxiety-dataset")
#%%
#car_file = os.path.join(path, "enhanced_anxiety_dataset.csv")
df_car =pd.read_csv("car_price_dataset.csv",delimiter=",")
df_car.head(2)
#%%
df_car.columns
#print("Path to dataset files:", path)
#%% kategorische in numerische one hot umwandeln - one hot encoding
df_car_dummies = pd.get_dummies(df_car, dtype= int,drop_first= True)
df_car_dummies.head(2)
df_car_dummies.shape
df_car_dummies.head(2)
# smoking_no und smoking_yes sind lineare doppelt kombinationen --> drop_first hat eine 0 oder 1 raus -- lineare kombinationen
# spaltenreihenfolge und merken und schauen welche erste wegfaellt.
# %% korrelationsmatrix bauen
numerical_features = df_car.select_dtypes(include=['int64','float64'])
corr = numerical_features.corr()
import seaborn as sns
# annot False laesst die korrelationskoeffizienten weg
sns.heatmap(corr, annot=True,cmap="coolwarm")
#%%
# sns.regplot(data=df_car_dummies, x= "Sleep Hours", y="Anxiety Level (1-10)", color="blue", line_kws={"color":"red"})
#%%
# Korrelationsmatrix hilft NUR für auswahl variablen mit linearem zusammenhang
# Anscomebe's quartet
# in .cache kagglehub laden jetzt saemtl datasets
# %% datensatz prep -- unBHÄNGIGE VON ABHÄNGIGE TRENNEN
# np.array -- keine spaltennamen mehr drin -- spaltennamen sind nur in df drin --
y = np.array(df_car_dummies[['Price']],dtype=np.float32)
y.shape
y
#%% X erklärende variablen
X = np.array(df_car_dummies.drop(columns=['Price']),dtype=np.float32)
X.shape
X
#%% data scaling
# gauss standard verteilung
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Scaled = scaler.fit_transform(X)
X_Scaled
# klassen werden immer mit großem buchstaben begenommen instanz ist klein buchstabe
# %%
# scikitlearn --
 