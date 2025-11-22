#%% packages
import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from enum import Enum
from pprint import pprint
load_dotenv(find_dotenv())
from pathlib import Path

os.getcwd()
os.chdir(Path(__file__).resolve().parent)
print("Working directory:", Path.cwd())

#%% Definiere eine Enum-Klasse für die möglichen Kategorien
class EmailCategory(str, Enum):
    AUSLAND = "Ausland"
    HILFSMITTEL = "Hilfsmittel"
    KRANKENGELD = "Krankengeld"
    KRANKENHAUS = "Krankenhaus"
    SONSTIGES = "Sonstiges"

class ComplexityCategory(str, Enum):
    COMPLEX = "Complex"
    NONCOMPLEX ="Non-Complex"

class EmailSorter(BaseModel):
    category: EmailCategory = Field(
        ...,
        description="Kategorie, in die die Email einzuordnen ist."
    )
    complexity: ComplexityCategory = Field(..., description="Jede E-Mail sollte als komplex oder nicht-komplex kategorisiert werden")

parser = PydanticOutputParser(pydantic_object=EmailSorter)

#%% prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "Du bearbeitest Emails und kannst sie entsprechenden Kategorien zuordnen. Die Kategorien findest du in der Formatbeschreibung. {format_instruction}"),
    ("user", "Email: {email_content}")
]).partial(format_instruction=parser.get_format_instructions())
prompt_template

#%% Modellinstanz erstellen
model = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
# model = ChatOllama(model="gemma3:4b", temperature=0)

#%% develop the chain
chain = prompt_template | model | parser

# %% Modellinferenz
# email_content = "Ich muss nach Südafrika reisen und brauche eine gültige Krankenversicherung."
# email_content = "Ich will abends Salat essen."
# res = chain.invoke(input={"email_content": email_content})
# pprint(res)
# %% load all input data
import json
path_json_files = "krankenkassen_emails_dataset"
all_json_files = os.listdir(path_json_files)
all_emails_list = []
for json_file in all_json_files:
    with open(os.path.join(path_json_files, json_file), "r", encoding="utf-8") as f:
        email_data = json.load(f)
        all_emails_list.append(email_data)

all_emails_list

# %% inference for all emails
for email in all_emails_list[:3]:
    email_content = email["email_text"]
    res = chain.invoke(input={"email_content": email_content})
    email["category"] = str(res.category)
    email["complexity"] = str(res.complexity)

#%%
all_emails_list[:3]