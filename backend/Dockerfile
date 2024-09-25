
FROM python:3.9-slim


WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu


COPY . .


EXPOSE 10000


CMD ["python", "server.py"]
