FROM python:3.12

ENV HOST=0.0.0.0
ENV LISTEN_PORT=8080

EXPOSE 8080

RUN apt-get update && apt-get install -y git

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR /app/
COPY ./streamlit_app.py /app/Langchain_Chatbot.py

CMD ["streamlit", "run", "Langchain_Chatbot.py", "--server.port", "8080"]

