FROM python:3.9-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY app.py app.py
COPY assets assets

FROM python:3.9-slim AS deployer
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /app /app
WORKDIR /app
CMD ["python", "app.py"]
