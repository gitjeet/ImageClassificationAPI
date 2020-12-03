FROM python:3.8
WORKDIR /usr/src/app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install Pillow==2.2.1
RUN pip install --upgrade tensorflow-hub

COPY . .
CMD ["python","app.py"]
