FROM python:3.6

ADD . /app
WORKDIR /app

ADD requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
