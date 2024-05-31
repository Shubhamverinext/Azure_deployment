import os
import logging
from logging.handlers import TimedRotatingFileHandler
from langchain.chat_models import AzureChatOpenAI
import json
import yaml
from firm_case_classifier_api_v8 import process_query
#from azure.identity import DefaultAzureCredential
from azure.identity import AzureCliCredential
from azure.keyvault.secrets import SecretClient
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
        # Key Vault URL
        key_vault_url = "https://caseratekeyvault.vault.azure.net/"
        # DefaultAzureCredential will handle authentication for managed identity, Azure CLI, and environment variables.
        #credential = DefaultAzureCredential()
        credential = AzureCliCredential()
        # Create a SecretClient using the Key Vault URL and credential
        client = SecretClient(vault_url=key_vault_url, credential=credential)
        # Retrieve a secret
        self.OPENAI_API_KEY = client.get_secret("pl-open-api-key").value
        self.OPENAI_DEPLOYMENT_VERSION = client.get_secret("pl-openai-deployment-version").value
        self.OPENAI_DEPLOYMENT_ENDPOINT = client.get_secret("pl-openai-deployment-endpoint").value
        self.OPENAI_DEPLOYMENT_NAME = client.get_secret("pl-openai-deployment-name").value
        self.OPENAI_MODEL_NAME = client.get_secret("pl-openai-model").value

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
                #return final_result
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
                        
                #return final_result
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
                #return final_result
            return final_result
        except Exception as error:
            logging.exception("An error occurred in flag_check: %s", error)
    except Exception as error:
        logging.exception("An error occurred in flag_check: %s", error)  


class caseClassifierApp:
    def __init__(self, case_classifier):
        self.case_classifier = case_classifier

    def send(self, msg):
        result = self.case_classifier.analyze_case(msg)
        return result
    
if __name__ == "__main__":

    while True:
        query = input('you: ')
        if query == 'q':
            break
        elif query.strip() == "":
            continue
        #response = app.send(query)
        response = flag_check(query)
        logging.info('Final result generated: %s', response)
        print("response", response)
        