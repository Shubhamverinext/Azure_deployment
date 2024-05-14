from langchain.document_loaders import DataFrameLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
#from dotenv import load_dotenv
import openai
import os
import pandas as pd
import sqlite3
import json

OPENAI_API_TYPE = "Azure"
OPENAI_API_KEY = "5f77f653b73e42948a920b9c2e6f76d8"
OPENAI_DEPLOYMENT_ENDPOINT = "https://pl-ver-openai.openai.azure.com/"
OPENAI_DEPLOYMENT_NAME = "gpt-35"
OPENAI_MODEL_NAME = "gpt-35-turbo"
OPENAI_API_VERSION = "2024-02-15-preview"


OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "embedding"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

DATA_PATH = "data/PL_Training_Data_New.xlsx"
DB_FAISS_PATH = 'vectorstores/pondlehocky'

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_DEPLOYMENT_ENDPOINT"] = OPENAI_DEPLOYMENT_ENDPOINT
os.environ["OPENAI_DEPLOYMENT_NAME"] = OPENAI_DEPLOYMENT_NAME
os.environ["OPENAI_MODEL_NAME"] = OPENAI_MODEL_NAME
os.environ["OPENAI_API_VERSION"] = OPENAI_API_VERSION

os.environ["OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME"] = OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME
os.environ["OPENAI_ADA_EMBEDDING_MODEL_NAME"] = OPENAI_ADA_EMBEDDING_MODEL_NAME


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
OPENAI_DEPLOYMENT_ENDPOINT = os.getenv("OPENAI_DEPLOYMENT_ENDPOINT")
OPENAI_DEPLOYMENT_NAME = os.getenv("OPENAI_DEPLOYMENT_NAME")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = os.getenv("OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME")
OPENAI_ADA_EMBEDDING_MODEL_NAME = os.getenv("OPENAI_ADA_EMBEDDING_MODEL_NAME")

#init Azure OpenAI
openai.api_type = OPENAI_API_TYPE
openai.api_base = OPENAI_DEPLOYMENT_ENDPOINT
openai.api_key = OPENAI_API_KEY
openai.api_version = OPENAI_API_VERSION

def fetch_feedback_data(start_date, end_date):
    
    # Connect to the database
    conn = sqlite3.connect('Cases.db')
    curr = conn.cursor()

    # Fetch data from the Cases table within the date range
    curr.execute("SELECT CaseDescription, FeedbackId FROM Cases WHERE Date >= ? AND Date <= ?", (start_date, end_date))
    rows = curr.fetchall()
    data_list = []
    # Iterate through each row of Cases table data
    for row in rows:
        # Fetch data from Feedback table corresponding to FeedbackId
        curr.execute("SELECT * FROM Feedback WHERE FeedbackId = ?", (row[1],))
        feedback_row = curr.fetchall()
        # Append description and feedback to the list
        if feedback_row:
            feedback_row = feedback_row[0]
            description = row[0]
            curr.execute("SELECT CaseType FROM PrimaryCaseTypes WHERE PrimaryCaseTypeId = ?", (feedback_row[1],))
            PrimaryCaseType = curr.fetchone()
        
            curr.execute("SELECT CaseType FROM SecondaryCaseTypes WHERE ScondaryCaseTypeId = ?", (feedback_row[2],))
            SecondaryCaseType = curr.fetchone()

            curr.execute("SELECT CastRating FROM Caserating WHERE CaseRatingId = ?", (feedback_row[3],))
            CaseRating = curr.fetchone()

            curr.execute("SELECT Name FROM CaseStates WHERE CaseStateId = ?", (feedback_row[4],))
            CaseState = curr.fetchone()

            # curr.execute("SELECT Name FROM HandlingFirms WHERE HandlingFirmId = ?", (feedback_row[5],))
            # HandlingFirm = curr.fetchone()

            TrainingData = description + "\nPrimary Case Type: " + PrimaryCaseType[0] + "\nSecondary Case Type: " + SecondaryCaseType[0] + "\nCase Rating: " + CaseRating[0] + "\nCase State: " + CaseState[0]
            data_list.append({'TrainingData': TrainingData})
 
            
    # Don't forget to close the connection when you're done
    conn.close()

    # Don't forget to close the connection when you're done
    conn.close()
    return data_list


if __name__ == "__main__":

    start_date = '2024-04-26'
    end_date = '2024-05-02'
    embeddings=OpenAIEmbeddings(deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
                                model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
                                openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
                                openai_api_type=OPENAI_API_TYPE,
                                model_kwargs = {'deployment_id': OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME},
                                chunk_size=1)
    
    data_list = fetch_feedback_data(start_date, end_date)
    df = pd.DataFrame(data_list)
    loader = DataFrameLoader(df, page_content_column="TrainingData")
    docsearchs = FAISS.load_local(DB_FAISS_PATH, embeddings)
    documents = loader.load() 
    docsearch = FAISS.from_documents(documents, embeddings)
    #docsearch.docstore._dict
    docsearchs.merge_from(docsearch)
    #docsearchs.docstore._dict
    docsearchs.save_local(DB_FAISS_PATH)

