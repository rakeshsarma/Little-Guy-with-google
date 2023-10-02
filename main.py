from dotenv import load_dotenv
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *



import os
import json

from langchain.memory.buffer_window import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory, ConversationKGMemory,CombinedMemory
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

llm_context = ChatOpenAI(temperature=0, model_name="gpt-4")
llm_code = ChatOpenAI(temperature=0, model_name="gpt-4")
'''
chat_history_buffer = ConversationBufferWindowMemory(
    k=5,
    memory_key="chat_history_buffer",
    input_key="input"
    )
   
chat_history_summary = ConversationSummaryMemory(
    llm=llm_context, 
    memory_key="chat_history_summary",
    input_key="input"
    )

chat_history_KG = ConversationKGMemory(
    llm=llm_context, 
    memory_key="chat_history_KG",
    input_key="input",
    input_variables = ["input", "agent_scratchpad"]
    )

memory = CombinedMemory(memories=[chat_history_buffer, chat_history_summary, chat_history_KG])
'''



db = SQLDatabase.from_uri(sqlalchemy_url, include_tables=["event_level_data"])



toolkit = SQLDatabaseToolkit(db=db, llm=llm_code)

agent_executor = create_sql_agent(
	llm=llm_code,
	toolkit=toolkit,
	verbose=True,
	top_k=100 
	)
#agent 2 with memory
'''
agent_executor = create_sql_agent(
	llm=llm_code,
	toolkit=toolkit,
	verbose=True,
	top_k=100,
    agent_executor_kwargs={"memory": memory},
    input_variables = ['input', 'agent_scratchpad','chat_history_KG','chat_history_summary','chat_history_buffer']
    
	)

'''

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
#answer = agent_executor.run("Can you show me the trends of psuedo users day by day. Are they increasing or decreasing?")
#agent_executor.run("Hi")
#answer = agent_executor.run("A day is called a good day if the no of psudo users acquired in a day is greater than the mean users in that week. How many good days did we have last week? Mention the dates")
#answer = agent_executor.run("which are the top 10 cities by revenue")
#answer = agent_executor.run("what are the 3 top channels of revenue for top 3 cities by revenue?")

#answer = agent_executor.run("give me top 10 three event sequences which leads to highest revenue ")
#answer = agent_executor.run("give me top 10 three event sequences which leads to highest revenue for users coming from  channels")
answer = agent_executor.run("Event sequences are set of 3 consecutive events orderd by date time. give me top 10 'three event sequences' which leads to highest revenue for users coming from Online Stamp Paper channel")
#answer = agent_executor.run("give me top 10 'three event sequences' which leads to highest revenue for users coming from Online Stamp Paper channel")


#print(answer)