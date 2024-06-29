FROM python:3.12.3

WORKDIR /app

RUN apt-get -y update && apt-get install -y \
  python3-dev \
  apt-utils \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade setuptools
RUN pip3 install \
    scikit-learn \
    streamlit \
    pandas

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "--server.port", "8501", "app.py"]
