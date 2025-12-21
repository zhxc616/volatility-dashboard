
FROM python:3.12-slim

# directory creation
WORKDIR /app

# dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

# open port
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]