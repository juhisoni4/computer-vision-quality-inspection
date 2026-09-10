# Base Image
FROM python:3.10.19


WORKDIR /usr/src/app

COPY / .

COPY train.py ./train.py

# Update System
RUN apt update

# Update pip
RUN pip install --upgrade pip

# Install requirements. Can be commented out if a requirements.txt does not exists.
# RUN pip install --no-cache-dir -r src/requirements.txt
RUN pip install --no-cache-dir -r src/requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python || true && \
    pip install --no-cache-dir --force-reinstall opencv-python-headless && \
    pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

RUN groupadd -r user && useradd -r -g user user

ENTRYPOINT [ "python3", "/usr/src/app/train.py" ]
