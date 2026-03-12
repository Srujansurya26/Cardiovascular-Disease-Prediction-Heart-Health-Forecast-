FROM python:3.7.9

WORKDIR /app

COPY . /app


RUN pip install --no-cache-dir -r requirements.txt

CMD ["streamlit", "run", "final_app.py", "--server.address=0.0.0.0"]