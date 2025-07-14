#%% packages
from langchain_community.llms import Ollama

#%%
model = Ollama(model="deepseek-r1:1.5b")

response = model.invoke("write python code for calculating prime numbers up to 10 numbers")

#%%
from pprint import pprint
pprint(response)

