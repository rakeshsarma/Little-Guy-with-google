
from dotenv import load_dotenv
import uvicorn
from datetime import datetime,timedelta
from google.cloud import bigquery
from sqlalchemy import *
from sqlalchemy.engine import create_engine
from sqlalchemy.schema import *




import os
import json
import requests

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


from typing import Annotated
from pybigquery.api import ApiClient

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel

from passlib.context import CryptContext
from jose import jwt, JWTError


SECRET_KEY = "35d819060d15fa47e7a3f93df39dc12d47ccefdac61afba6e31765cf68f14dcc"
ALGORITHM = "HS256"  #openssl rand -hex 32
ACCESS_TOKEN_EXPIRE_MINUTES = 30




db = {
    "tim" :{
        "username":"tim",
        "full_name" : "Tim Rex",
        "email" : "tim@gmail.com",
        "hashed_password" : "$2b$12$Iz.M6B/jGZ9KfcuE0R3Wf.QaZPvpVQdoNFO.2SaRwJRZK9iI9ye6m",
        "disabled": False
    }
}



class Token(BaseModel):
    access_token : str
    token_type : str

class TokenData(BaseModel):
    username : str or None = None

class User(BaseModel):
    username:str
    email:str or None = None
    full_name :str or None = None
    disabled : bool or False

class UserInDB(User):
    hashed_password : str
    
pwd_context = CryptContext(schemes=["bcrypt"], deprecated ="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl = "token")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers=["*"]
)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db,username:str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(db, username:str, password: str):
    user = get_user(db, username)   
    if not user:
        return False  
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data : dict, expires_delta : timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now() + expires_delta
    else:
        expire = datetime.now() + timedelta(minutes = 15)
    to_encode.update({"exp":expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def authenticateBubbleUser(bubbleUsername, bubblePassword, version):
    prod_url = "https://qseo.quantacus.ai/api/1.1/wf/googleAuthenticator"
    test_url = "https://qseo.quantacus.ai/version-test/api/1.1/wf/googleAuthenticator"
    payload = {"username": bubbleUsername, "password":bubblePassword}
    
    if(version =="prod"):
        response = requests.get(prod_url, params=payload)
    else:
        response = requests.get(test_url, params=payload)
    
    print(response)

    return response


async def get_current_user(token:str= Depends(oauth_2_scheme)):
    credential_exception = HTTPException(status_code= status.HTTP_401_UNAUTHORIZED, detail ="could not validate the credentails.", headers ={"WWW-Authenticate":"Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username : str = payload.get("sub")
        if username is None:
            raise credential_exception
        token_data = TokenData(username = username)
    except JWTError:
        raise credential_exception
    user = get_user(db, username = token_data.username)
    if user is None:
        raise credential_exception
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive User")
    return current_user

@app.post("/token", response_model = Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(db, form_data.username,form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail= "incorrect username or password",headers ={"WWW-Authenticate":"Bearer"} )
    access_token_expires = timedelta(minutes = ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data = {"sub" : user.username}, expires_delta = access_token_expires)
    return {"access_token": access_token, "token_type" :"bearer"}

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



#llm_context = ChatOpenAI(temperature=0, model_name="gpt-4")
#llm_code = ChatOpenAI(temperature=0, model_name="gpt-4")


#using GPT 3.5 turbo for validation
llm_context = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_code = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")



'''
#commenting memory piece, will be picked up later
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



database = SQLDatabase.from_uri(sqlalchemy_url, include_tables=["event_level_data"])



toolkit = SQLDatabaseToolkit(db=database, llm=llm_code)

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
#answer = agent_executor.run("Event sequences are set of 3 consecutive events orderd by date time. give me top 10 'three event sequences' which leads to highest revenue for users coming from Online Stamp Paper channel")
#answer = agent_executor.run("give me top 10 'three event sequences' which leads to highest revenue for users coming from Online Stamp Paper channel")

#answer = agent_executor.run("Give me the payment funnel for any user using this app. Only consider users who have purchased to make this funnel")

#print(answer)


@app.get("/ask")
def answer (question :str):
    answer = agent_executor.run(question)
    return answer


@app.get("/ask/auth")
def answer2 (question :str, current_user: User = Depends(get_current_active_user)):
    answer = agent_executor.run(question)
    return answer





@app.get("/bubble/auth")
def  answer3 (bubbleUsername :str, bubblePassword: str, version: str):
    print (bubbleUsername)
    print (bubblePassword)
    print (version)
    response = authenticateBubbleUser(bubbleUsername, bubblePassword, version)
    return response
    
#pwd = get_password_hash("12345")    
#print(pwd)

