**Framework for evaluation and running MIC21 models**
========

The framework is based on FiftyOne (an open-source tool for building high-quality datasets and computer vision models), Yolact, Detectron2 and is an implementation of Mask R-CNN on Python3, Keras and TensorFlow. We pre-trained Fast R-CNN models using Detectron2 framework with ground truth annotations for 130 subdomains in the Sport, Transport, Arts and Security thematic fields and this resulted in 130 models which generate bounding boxes and segmentation masks for each instance of an object in an image. It's based on ResNet 101 backbone.

The repository includes:
1. Source code of Mask R-CNN built on FPN and ResNet101.
2. Training code for MS COCO.
3. Pre-trained weights for MS COCO and above 800 labels.
4. Jupyter notebooks to visualize the detection pipeline at every step.
5. Evaluation on MS COCO metrics integrated in FiftyOne

**System requirements**

CUDA 11.3 capable graphic controller.

**Installation**

Clone the git repository for the framework:

```git clone https://github.com/DCL-IBL/mic21-framework.git```

Navigate to the top directory of the mic21 framework:

```cd mic21-framework```

Download [mic21 model images](https://dcl.bas.bg/MIC-21/model_images/) together with annotations in JSON into `/server/uploads` subdirectory 

```wget -r -np -k https://dcl.bas.bg/MIC-21/model_images/ -P ./server/uploads```

Download [mic21 model weights](https://dcl.bas.bg/MIC-21/models/) into `/work/output` subdirectory.

```wget -r -np -k https://dcl.bas.bg/MIC-21/models/ -P ./work/output```

Setup the docker to host port mapping according to your preferences by modififying `docker-compose.yml`

```
version: "2"
services:
  web:
    build: .
    ports:
      - "PORT_DEVL:8888"
      - "PORT_SERV:5000"
      - "PORT_VIEW:5151"
    volumes:
      - ../:/host
    runtime: nvidia
```

where `PORT_DEVL` is the port for accessing jupyter notebooks used for training and accessing the models, `PORT_SERV` is the port for accessing the framework programming interface and the `PORT_VIEW` is used for accessing the visualization and presentation platform FiftyOne. The development port (PORT_DEVL) is optional and might be kept closed. 

Navigate to `/host/yolact` directory and download Yolact model weights used for prediction with:

```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UYy3dMapbH1BnmtZU4WH1zbYgOzzHHf_" -O yolact_base_54_800000.pth && rm -rf /tmp/cookies.txt
```

Build the docker image with

```docker-compose build```

Start the docker image with

```docker-compose up```

**API server interface**

All functions are accessed through http://domain:srv_port?path

|Path | Function|
|-----|---------|
|predict?model=yolact&categ_name=categ&threshold=0.9|Prediction of annotations using the Yolact software. Replace the categ with the categ name you intend to make a prediction on. The images from the category have to be uploaded in the server/uploads/categ folder. The threshold value can be modified in the range from 0 to 1.|