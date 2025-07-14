#%% packages
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
load_dotenv(find_dotenv(usecwd=True))

#%% model

#%% pydantic model
class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    steps: list[str] = Field(description="The steps to solve the problem")
    
#%% prompt
def self_consistency_cot(user_input: str, n_rounds: int = 3, model_name: str = "llama-3.1-8b-instant"):
    answers = []
    prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant that can answer questions. Think step by step. Please think step-by-step! Answer the question in JSON format with the following fields: answer, steps. The answer comes from the last step."),
            ("human", "{user_input}")
        ])
    model = ChatGroq(model=model_name, 
                 temperature=0.5)

    # chain
    chain = prompt_template | model | SimpleJsonOutputParser(pydantic_object=Answer)
    task = {
        "user_input": user_input
    }

    for _ in range(n_rounds):
        res = chain.invoke(task)
        answers.append(res)
    return answers
#%%
task = "When Michael was 9 years old, his brother was two-thirds his age. If Michael is currently 40 years old, how old is his brother now? Please think step-by-step!"
res = self_consistency_cot(user_input=task, model_name="llama3-70b-8192", n_rounds=3)
# %%
res
# %%
