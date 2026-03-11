FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .
RUN sed -i 's/sklearn/scikit-learn/g' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

