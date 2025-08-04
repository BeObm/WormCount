FROM python:3.10
LABEL authors="bolou"

WORKDIR /code
COPY ./requirements.txt /code/requirements.txt