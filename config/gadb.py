from sqlalchemy import create_engine
import sqlalchemy
from google.cloud.sql.connector import Connector

#db_password = acces_secret_version()
connector = Connector()
def getconn():
    conn=connector.connect(
        "ga4-bq-connector:asia-south1:littleguy-mysql",
        "pymysql",
        user="rakeshsarma",
        password = "1@Superman",
        db= "Quantacus"
    )
    return conn

engine = create_engine(
    "mysql+pymysql://",
    creator = getconn
)

#insert_statement = sqlalchemy.text("INSERT INTO Quantacus.user (BubbleID,GCPID,PersonID) VALUES (:BBcode, :GCPcode, '1')")

def insert_user(BBcode:str, GCPcode :str):
    insert_statement = sqlalchemy.text("INSERT INTO Quantacus.user (BubbleID,GCPID,PersonID) VALUES (:BBcode, :GCPcode, '1')")
    with engine.connect() as db_conn:
        db_conn.execute(insert_statement, BBcode = BBcode, GCPcode =GCPcode)

def select_user(GCPcode :str):
    
    select_statement = sqlalchemy.text("select GCPID from Quantacus.user where GCPID in (:GCPcode)")
    with engine.connect() as db_conn:
        result = db_conn.execute(select_statement, GCPcode = GCPcode).fetchone()
        return result


#test_id = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhYnpzZVc0OFY1QUVHYTQzYm9lWHFZb3c9PWMifQ.DTuLgT08PffjGnmcxEmxZerVoVmZDovSvbcWGRefhw8'
#print(select_user(test_id))

