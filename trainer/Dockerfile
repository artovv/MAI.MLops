FROM python:3.10

WORKDIR /app

COPY train.py .

RUN pip install mlflow scikit-learn

CMD ["python", "train.py"]