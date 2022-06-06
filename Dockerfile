FROM nvidia/cuda:11.3.0-base-ubuntu18.04
RUN apt-get upgrade && apt-get -y update
RUN apt-get install -y build-essential python3.7 python3-pip python3-dev
RUN pip3 -q install pip --upgrade
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN pip3 install jupyter
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y python3-opencv
RUN pip3 install opencv-python
RUN apt-get install -y git
RUN pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
RUN git clone https://github.com/facebookresearch/detectron2.git
RUN python3 -m pip install -e detectron2
RUN pip3 install flask
RUN pip3 install flask-restful
RUN apt-get install -y curl
RUN curl -fsSL https://deb.nodesource.com/setup_17.x | bash -
RUN apt-get install -y nodejs
RUN pip3 install mongo
RUN pip3 install fiftyone
RUN pip3 install asyncio
#RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.1/install.sh | bash
#RUN nvm install node
RUN npm install -g yarn
RUN apt-get install -y libcurl4 openssl locales
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen
RUN apt-get install -y netcat
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
WORKDIR .
EXPOSE 8888
EXPOSE 5000
EXPOSE 5151
COPY cmd_wrapper_script.sh .
CMD ["./cmd_wrapper_script.sh"]