**Framework for evaluation and running MIC21 models**
========

The framework is based on FiftyOne (an open-source tool for building high-quality datasets and computer vision models), Yolact, Detectron2 and is an implementation of Mask R-CNN on Python3, Keras and TensorFlow. We pre-trained Fast R-CNN models using Detectron2 framework with ground truth annotations for 130 subdomains in the Sport, Transport, Arts and Security thematic fields and this resulted in 130 models which generate bounding boxes and segmentation masks for each instance of an object in an image. It's based on ResNet 101 backbone.

The repository includes:
1. Source code of Mask R-CNN built on FPN and ResNet101.
2. Training code for MS COCO.
3. Pre-trained weights for MS COCO and above 800 labels.
4. Jupyter notebooks to visualize the detection pipeline at every step.
5. Evaluation on MS COCO metrics integrated in FiftyOne

**Evaluation and running MIC21 models in fiftyone

**MIC21 models**

MIC21 models are pretrained  ……….

**Datasets**

	Download MIC21 models  from …. and ….

**Installation**

Clone the git repository for the framework:

```git clone https://github.com/DCL-IBL/mic21-framework.git```

Navigate to the top directory of the mic21 framework:

```cd mic21-framework```

Download [mic21 model images](https://dcl.bas.bg/MIC-21/model_images/) together with annotations in JSON into `/server/uploads` subdirectory 

```wget -r -np -k https://dcl.bas.bg/MIC-21/model_images/ -P ./server/uploads```


docker-compose up

