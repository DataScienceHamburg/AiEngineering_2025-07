#%% packages
from langchain import hub
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv('.env')
from pprint import pprint

#%% fetch prompt
prompt_template = hub.pull("hardkothari/prompt-maker")

#%% get input variables
prompt_template.input_variables

# %% model
model = ChatGroq(model="llama3-70b-8192", 
                 temperature=0)

# %% chain
chain = prompt_template | model | StrOutputParser()

# %% invoke chain
lazy_prompt = "Harvard Professor, 60 years old, Astronomy"
task = "Write a short bio"
improved_prompt = chain.invoke({"lazy_prompt": lazy_prompt, "task": task})
# %%
print(improved_prompt)

# %% run model with improved prompt
res = model.invoke(improved_prompt)
print(res.content)

# %%
res = model.invoke(f"{lazy_prompt}, {task}")
print(res.content)
# %%
