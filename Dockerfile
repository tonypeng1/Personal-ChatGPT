FROM python:3.11-slim-bullseye

LABEL project="personal-chatgpt"  # Add metadata labels

WORKDIR /app

# RUN apt-get update && apt-get install -y git

COPY requirements.txt /app
RUN pip3 install -r requirements.txt

COPY . /app

EXPOSE 8501
ENTRYPOINT ["streamlit", "run", "personal-chatgpt/personal_chatgpt.py", "--server.port=8501", "--server.address=0.0.0.0"]