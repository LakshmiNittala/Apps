
RUN apt-get update && apt-get install -y git

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR /app/
COPY ./streamlit_app.py /app/streamlit_app.py

CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8080"]
