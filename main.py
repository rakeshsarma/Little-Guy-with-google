from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

import os
import json

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms.openai import OpenAI
from langchain.agents import AgentExecutor

with open('keys.json') as json_file:
    api_key = json.load(json_file)['api_key']
os.environ["OPENAI_API_KEY"] = api_key

service_account_file = "ga4-bq-connector-dcaae8e68326.json" # Change to where your service account key file is located

project = "ga4-bq-connector"
dataset = "analytics_399638956"
table = "events_20230912"

sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'

db = SQLDatabase.from_uri(sqlalchemy_url)

llm = OpenAI(temperature=0, model="text-davinci-003")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
	llm=llm,
	toolkit=toolkit,
	verbose=True,
	top_k=5,
	)

print(agent_executor.run("How many unique event_Date are there in the table?"))