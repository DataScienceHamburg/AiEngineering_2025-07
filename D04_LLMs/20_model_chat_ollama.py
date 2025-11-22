import os
from langchain_ollama import ChatOllama
 
#%%
model = ChatOllama(model="gemma2:2b",temperature=0, top_p=0.8 ,top_k=3) # ChatGroq holt sich api key automatisch aus .env muss nicht extra verweisen
#%% Model
user_query = "Was sind die besten drei zero-shot classification modesle, die mit deutsch arbeiten."
res = model.invoke(user_query)
#%%
from pprint import pprint
pprint(res.content)
#%%