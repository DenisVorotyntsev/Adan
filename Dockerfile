FROM ubuntu:20.04

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip
RUN pip install notebook
RUN pip install --upgrade pip

RUN mkdir /work
COPY . /work
WORKDIR /work

RUN pip3 install --user -e .
RUN pip3 install -r requirements_tests.txt