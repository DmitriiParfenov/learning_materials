FROM python:3.12-slim

WORKDIR /mlflow_app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /mlflow_app/requirements.txt
RUN pip install --no-cache-dir -r /mlflow_app/requirements.txt
COPY . /mlflow_app
