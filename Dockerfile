FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu22.04
RUN apt update -y
RUN apt install -y python3-pip python3-venv
WORKDIR /app
RUN mkdir /app/scores
RUN mkdir /app/output
COPY *.whl .
RUN python3 -m venv venv
RUN venv/bin/pip install *.whl
COPY config.json .
CMD ["venv/bin/python", "-m", "aaai2023.main", "-f", "config.json"]
