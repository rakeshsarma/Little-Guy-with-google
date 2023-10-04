from sqlalchemy import create_engine, MetaData
from google.cloud.sql.connector import Connector

#db_password = acces_secret_version()
connector = Connector()
def getconn():
    conn=connector.connect(
        "ga4-bq-connector:asia-south1:littleguy-mysql",
        "pymysql",
        user="root",
        password = "MqbY+/?odFA;}%uf",
        db= "gcp_demo"
    )
    return conn

engine = create_engine(
    "mysql+pymysql://",
    creator = getconn
)

meta = MetaData()
conn = engine.connect()