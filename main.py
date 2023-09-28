from dotenv import load_dotenv
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *


import os
import json

from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentExecutor
from sqlalchemy import *

from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *

from pybigquery.api import ApiClient



#loading the environment
load_dotenv('allkeys.env')

openai_api_key = os.getenv("OPENAI_API_KEY")


service_account_file = "service-account-key.json" # Change to where your service account key file is located

project = "ga4-bq-connector"
dataset = "test_dataset_id"
#table = "EVENTS_UNION"
table = "event_level_data"

#dataset = "analytics_254245242"
#table = "events_*"


sqlalchemy_url = f'bigquery://{project}/{dataset}?credentials_path={service_account_file}'

print (sqlalchemy_url)


db = SQLDatabase.from_uri(sqlalchemy_url, include_tables=["event_level_data"])

#llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent_executor = create_sql_agent(
	llm=llm,
	toolkit=toolkit,
	verbose=True,
	top_k=10,
	)

#agent_executor.run("give me the percentage distribution of revenue by channel for yesterday. if there's in none take it as 0. Use IOUX_PARAMS_EVENT_SESSION ")
#agent_executor.run("list down all the channels in the dataset?")
#agent_executor.run("How many channels are there in the dataset?")
#agent_executor.run("describe the dataset?")
#agent_executor.run("Count the rows by date?")
#agent_executor.run("compare the percentage distribution of purchase revenue by channel  for yesterday and the day before yesterday. Highlight the increase and decrease contribution for each channel. ")
#agent_executor.run("What is today's date? how many psuedo users did we acquire yesterday? Query event_level_data table")
#agent_executor.run("what was the total revenue yesterday? Query event_level_test table")
#agent_executor.run("what is the amount by all the channels distribution this week look like? Query event_level_test table")
#agent_executor.run("what is the sum of amount? Query event_level_test table")
#gent_executor.run("what is the sum of amount day before yesterday? Query event_level_data table")
#agent_executor.run("what is the sum of amount yesterday? Query event_level_data table")
#agent_executor.run("How many active psuedo users do we have this month? Query event_level_data table")
#agent_executor.run("How many active psuedo users do we have in the month of september? Query event_level_data table")
#agent_executor.run("How many active psuedo users do we have in this month? Query event_level_data table")

#agent_executor.run("which channel performed better this week and why?")
#agent_executor.run("what are the top 3 channels and what is their contibution by percent to the revenue?")
agent_executor.run("Can you show me the trends of psuedo users day by day")

