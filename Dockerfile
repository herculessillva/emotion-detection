# Get build image arg
ARG IMAGE
# Base image
FROM $IMAGE

# Get build args
ARG ZIP_ID

ENV TZ=America/Fortaleza
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies and python3
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx libgtk2.0-dev libpq-dev gcc
RUN apt-get -y install git cmake wget unzip curl
# RUN apt-get -y install python3
# RUN apt-get -y install python3-dev
# RUN apt-get -y install python3-pip

COPY . /usr/src/app

# Download Emoction detector files
WORKDIR /usr/src/app/lib
RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='${ZIP_ID} -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=${ZIP_ID}" -O conf.zip && rm -rf /tmp/cookies.txt
RUN unzip conf.zip -d ./
RUN rm -rf conf.zip

# Return to root path
WORKDIR /usr/src/app

RUN chmod +x setup-python3.8.sh
RUN sh setup-python3.8.sh

# Compile DLIB
RUN git clone https://github.com/davisking/dlib.git
WORKDIR dlib
RUN mkdir build
WORKDIR build
RUN cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
RUN cmake --build .
WORKDIR ..
RUN python3 setup.py install

# Return to root path
WORKDIR /usr/src/app

# Install project requirements
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
RUN pip3 install fastapi
RUN pip3 install uvicorn

EXPOSE 8083

# Run service
CMD ["python3", "main.py"]
