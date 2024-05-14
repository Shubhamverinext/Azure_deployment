from flask import Flask, render_template, request, redirect, url_for, session
import re
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import os
from functools import wraps
import openai
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import pandas as pd
import requests
import smtplib
from email.message import EmailMessage
import logging
from logging.handlers import TimedRotatingFileHandler
import yaml
from firm_case_classifier_api_v8 import process_query

# Configure logging with a TimedRotatingFileHandler
# Configure logging with a TimedRotatingFileHandler
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        TimedRotatingFileHandler(
            filename='app.log',
            when='W0',  # Rotate logs on a weekly basis, starting on Monday
            backupCount=1  # Retain one backup log file (the current week's log)
        )
    ],
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class caseClassifier:
    def __init__(self):
        # Load the YAML file
        with open('config.yml', 'r') as yaml_file:
            config = yaml.safe_load(yaml_file)

        # Access the configuration values
        self.OPENAI_API_KEY = config['BaseConfig']['OPENAI_API_KEY']
        self.OPENAI_DEPLOYMENT_VERSION = config['BaseConfig']['OPENAI_DEPLOYMENT_VERSION']
        self.OPENAI_DEPLOYMENT_ENDPOINT = config['BaseConfig']['OPENAI_DEPLOYMENT_ENDPOINT']
        self.OPENAI_DEPLOYMENT_NAME = config['BaseConfig']['OPENAI_DEPLOYMENT_NAME']
        self.OPENAI_MODEL_NAME = config['BaseConfig']['OPENAI_MODEL_NAME']

        os.environ["OPENAI_API_KEY"] = self.OPENAI_API_KEY
        os.environ["OPENAI_DEPLOYMENT_ENDPOINT"] = self.OPENAI_DEPLOYMENT_ENDPOINT
        os.environ["OPENAI_DEPLOYMENT_NAME"] = self.OPENAI_DEPLOYMENT_NAME
        os.environ["OPENAI_MODEL_NAME"] = self.OPENAI_MODEL_NAME
        os.environ["OPENAI_DEPLOYMENT_VERSION"] = self.OPENAI_DEPLOYMENT_VERSION 

        self.custom_prompt_template = """
        Please determine if the following description constitutes a legal case: "{question}"
        Your task is to analyze the provided description and indicate whether it resembles a legal case
        Consider elements such as parties involved, legal issues, relevant laws
        court proceedings, or any other factors that typically define a legal case
        If the description aligns with what you understand as a legal case, 
        The Output should be strictly in JSON format and the JSON structure must have the following key values:
        "Status" : "YES or NO if description is legal say YES if not than say NO",
        "Explanation" : "Explanation for given response"

        """

    def load_llm(self):
        llm = AzureChatOpenAI(
            deployment_name=self.OPENAI_DEPLOYMENT_NAME,
            model_name=self.OPENAI_MODEL_NAME,
            openai_api_base=self.OPENAI_DEPLOYMENT_ENDPOINT,
            openai_api_version=self.OPENAI_DEPLOYMENT_VERSION,
            openai_api_key=self.OPENAI_API_KEY,
            openai_api_type="azure"
        )
        return llm
    
    def get_predictions(self, prompt_hf):
        llm = self.load_llm()
        predictions = llm.predict(prompt_hf)
        return predictions
    
    def analyze_case(self, query):
        try:
            hf_prompt = self.custom_prompt_template.format(question=query)
            predictions = self.get_predictions(hf_prompt)
            return predictions
        except Exception as error:
            print(error)
            return None  # Unable to determine the status from LLM  


class caseClassifierApp:
    def __init__(self, case_classifier):
        self.case_classifier = case_classifier

    def send(self, msg):
        result = self.case_classifier.analyze_case(msg)
        return result

def flag_check(query):
    case_classifier = caseClassifier()
    app = caseClassifierApp(case_classifier)
    response = app.send(query)
    try:
        pred = json.loads(response)
        print(pred.get("Explanation", "").strip())
        try:
            flag = pred.get("Status", "").strip().upper() == "YES"
            if flag is True:
                logging.info("This appears to be a legal case.")
                final_result = process_query(query)
                return final_result
            elif flag is False:
                logging.info("This does not appear to be a legal case.")
                final_result = str({
                            "PrimaryCaseType": " ",
                            "SecondaryCaseType": " ",
                            "CaseRating": " ",
                            "Case State" : " ",
                            "Is Workers Compensation (Yes/No)?": " ",
                            "Confidence(%)": " ",
                            "Explanation": pred.get("Explanation", "").strip(),
                            "Handling Firm" : "Unknown"
                        })
                        
                return final_result
            else:
                logging.warning("Unable to determine if it's a legal case due to an unexpected response.")
                final_result = '''
                    {
                        "PrimaryCaseType": " ",
                        "SecondaryCaseType": " ",
                        "CaseRating": " ",
                        "Case State" : " ",
                        "Is Workers Compensation (Yes/No)?": " ",
                        "Confidence(%)": " ",
                        "Explanation": "There is some error occured while answering your question, Please try with same case description again.  Sorry for an inconvenience Caused",
                        "Handling Firm" : "Unknown"
                    }
                    '''
                return final_result

        except Exception as error:
            logging.exception("An error occurred in flag_check: %s", error)
    except Exception as error:
        logging.exception("An error occurred in flag_check: %s", error) 
        
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
# Hardcoded root user credentials
ROOT_USERNAME = 'admin'
ROOT_PASSWORD = 'pladmin@123'
# Login route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the user is the hardcoded root user
        if username == ROOT_USERNAME and password == ROOT_PASSWORD:
            # Root user logged in successfully
            session['user_id'] = 1  # You can set any user ID for the root user
            return redirect(url_for('home'))
        else:
            return render_template('login.html',
                                   error='Invalid username or password. Please try again.'
                                   )
    return render_template('login.html')


# Logout route
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            print("User not logged in. Redirecting to index.")
            return redirect(url_for('index'))
        return f(*args, **kwargs)
    return decorated_function
# submit route
@app.route("/submit", methods=["GET", "POST"])
@login_required
def submit():
    if request.method == "POST":
        user_query = request.form.get("description")
        #input_json = {"msg": user_query}

        response = flag_check(user_query)
        print("Input description :", user_query)
        print(response)
        response_dict = json.loads(response)
        if user_query:
            # Extract data from the response_dict
            handling_firm = response_dict["Handling Firm"]
            primary_case_type = response_dict.get("PrimaryCaseType")
            secondary_case_type = response_dict.get("SecondaryCaseType")
            confidence = response_dict.get("Confidence(%)")
            explanation = response_dict.get("Explanation")
            case_rating = response_dict.get("CaseRating")
            Is_WC = response_dict.get("Is Workers Compensation (Yes/No)?")
            case_state = response_dict.get("Case State")
            #l= [primary_case_type,secondary_case_type,confidence,explanation,case_rating,handling_firm]
            #print("response_final//////",l)
        return render_template("test6.html", primary_case_type=primary_case_type,
                               secondary_case_type=secondary_case_type,
                               case_rating=case_rating,
                               Is_WC=Is_WC,
                               confidence=confidence,
                               explanation=explanation,
                               user_input=user_query,
                               case_state = case_state,
                               handling_firm = handling_firm,
                               result=response
                               )
    return render_template("test6.html")
# home route
@app.route("/home", methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        user_query = request.form.get("description")
        input_json = {"msg": user_query}
        response = api_fetch(api_url, input_json)
        print("Input description :", user_query)
        print(response)
        response_dict = json.loads(response)
        # if response is None:
        #     return render_template("test6.html", message="Response is None. Unable to load JSON.")
        # try:
        #     response_dict = json.loads(response)
        # except json.JSONDecodeError as e:
        #     return render_template("test6.html", message=f"Error decoding JSON: {e}")
        if user_query:
            handling_firm = response_dict["Handling Firm"]
            primary_case_type = response_dict.get("PrimaryCaseType")
            secondary_case_type = response_dict.get("SecondaryCaseType")
            confidence = response_dict.get("Confidence(%)")
            explanation = response_dict.get("Explanation")
            case_rating = response_dict.get("CaseRating")
            Is_WC = response_dict.get("Is Workers Compensation (Yes/No)?")
            case_state = response_dict.get("Case State")
            # Creating a dictionary with the data
            data = {
                "Description": [user_query],
                "Primary Case Type": [primary_case_type],
                "Secondary Case Type": secondary_case_type,
                "Case Rating": [case_rating],
                "Is WC": [Is_WC],
                "Case state": [case_state],
                "Handling Firm":[handling_firm],
                "Confidence": [confidence],
                "Explanation": [explanation],
            }
            # Creating a DataFrame from the dictionary
            df = pd.DataFrame(data)
            # Appending the data to the existing CSV file or create a new file if it doesn't exist
            df.to_csv("correct_case_data.csv", mode="a", index=False, header=not os.path.exists("correct_case_data.csv"))
            case_tier = re.search(r'\d+', case_rating).group()
            if int(case_tier) == 5 or int(case_tier) == 4:
                msg = case_tier
                print(msg)
                body = "There is a high case rating " + case_tier + "."
                #email_alert("Alert", body, "kadamkaran12345@gmail.com")
                message = "High case rating detected. Check your email for an alert."
                email_sent = True
        else:
            primary_case_type = ''
            secondary_case_type = ''
            confidence = ''
            explanation = ''
            case_rating = ''
            Is_WC = ''
            case_state = ''
            handling_firm = ''
        return render_template("test6.html",
                               primary_case_type=primary_case_type,
                               secondary_case_type=secondary_case_type,
                               case_rating=case_rating,
                               Is_WC=Is_WC,
                               confidence=confidence,
                               explanation=explanation,
                               user_input=user_query,
                               result=response,
                               message=message,
                               email_sent = email_sent,
                               case_state = case_state,
                               handling_firm = handling_firm
                               )
    return render_template("test6.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)