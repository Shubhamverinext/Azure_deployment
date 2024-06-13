from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import json
import logging
import uuid
from uvicorn import Config, Server
from pydantic import BaseModel
from Insert_feedback import feedback_table
#from azure.identity import DefaultAzureCredential
# from azure.identity import AzureCliCredential
# from azure.keyvault.secrets import SecretClient

# # Key Vault URL
# key_vault_url = "https://caseratekeyvault.vault.azure.net/"

# # DefaultAzureCredential will handle authentication for managed identity, Azure CLI, and environment variables.
# #credential = DefaultAzureCredential()
# credential = AzureCliCredential()

# # Create a SecretClient using the Key Vault URL and credential
# client = SecretClient(vault_url=key_vault_url, credential=credential)

# # Creating a FastAPI application instance
# app = FastAPI()

# # Defining OAuth2 security
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # Secret key and algorithm for JWT token
# #SECRET_KEY = client.get_secret("pl-secretkey-jwt").value
# #ALGORITHM = client.get_secret("pl-algorithm-jwt").value
# #ACCESS_TOKEN_EXPIRE_MINUTES = client.get_secret("pl-accesstoken-exp-min").value
SECRET_KEY = "83daa0256a2289b0fb23693bf1f6034d44396675749244721a2b20e896e11662"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 100

#database for storing user information
db = {
    "admin": {
        "username": "admin",
        "full_name": "Admin",
        "email": "admin@example.com",
        "hashed_password": "$2b$12$j5fnrZhC5BjTepx3yi5whO7d6AdWO3Zmynfb828EBdRVjBzGas/Km",  # hashed password for "admin@123"
        "disabled": False
    }
}

# Define the Token model
class Token(BaseModel):
    access_token: str
    token_type: str

# Define the TokenData model
class TokenData(BaseModel):
    username: str

# Define the User model
class User(BaseModel):
    username: str
    email: str = None
    full_name: str = None
    disabled: bool = None


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# Function to authenticate user
def authenticate_user(username: str, password: str):
    user = db.get(username)
    if not user or not verify_password(password, user["hashed_password"]):
        return False
    return user


# Function to create access token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# Function to verify password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        token_data = TokenData(username=username)
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.get(username)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


# Route to generate access token
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


# Endpoint to receive JSON data with OAuth2 authentication
@app.post("/receive_json")
async def receive_json(json_data: dict, current_user: dict = Depends(get_current_user)):
    try:
        generated_uuid = uuid.uuid4()
        print("generated_uuid_",generated_uuid)
        print("RESIVED JSON DATA",type(json_data))
        #json_data_db = json_data.dict()
        feedback_table(json_data, generated_uuid)
        print(".......................database")
        with open('log_file.json', 'a') as file:
            file.write(json.dumps(json_data) + '\n')
        return {"message": "JSON data received and saved to log file."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Entry point of the script
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    config = Config(app, host='0.0.0.0', port=80, log_level='info')
    server = Server(config)
    server.run()



