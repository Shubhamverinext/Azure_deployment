# Dockerfile
# Using the official Python image as a base image
FROM python:3.11

# Setting the working directory in the container
WORKDIR /Deployment
RUN apt-get update

# Copying the modified jose.py file into the container
COPY ./dist-packages/jose.py  /usr/local/lib/python3.11/site-packages/

# Copying the FastAPI application code into the container
COPY . .
COPY api_auth.py .

# Installing FastAPI, Uvicorn, and any other dependencies
COPY requirements.txt .


RUN pip3 install -r requirements.txt
RUN pip3 install passlib


# Exposing port 8080 to the outside world
EXPOSE 8080


CMD ["python3","api_auth.py"]


