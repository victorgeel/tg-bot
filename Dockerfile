FROM python:3.10.6-slim AS base

RUN apt-get update -y && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 8080

CMD python main.py

