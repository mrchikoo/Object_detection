<div align="center">

# Real-Time Hand-Gesture Object Detection

**Collect, label, train and run a custom TensorFlow Object Detection model on webcam gestures.**

![Python](https://img.shields.io/badge/Python-3.9-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Webcam-5C3EE8?logo=opencv&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

</div>

## Overview

An end-to-end workflow, split across two notebooks, for training a custom detector with the **TensorFlow 2 Object Detection API**. It recognises four hand gestures from a webcam feed: `ThumbsUp`, `ThumbsDown`, `ThankYou` and `LiveLong`.

## How it works

**1. `img_coll.ipynb` - data collection and labelling**
- Captures 5 webcam images per gesture with OpenCV into `Tensorflow/workspace/images/collectedimages/<label>`.
- Clones and launches [labelImg](https://github.com/tzutalin/labelImg) so you can draw bounding boxes (Pascal VOC XML).
- Archives the train/test image folders.

**2. `training.ipynb` - training and inference**
- Sets up the workspace, clones `tensorflow/models`, compiles the protobufs and installs the Object Detection API.
- Downloads the pretrained **SSD MobileNet V2 FPNLite 320x320 (COCO17)** checkpoint.
- Writes `label_map.pbtxt` and generates `train.record` / `test.record` with `generate_tfrecord.py`.
- Edits `pipeline.config` programmatically (4 classes, batch size 4, fine-tune from the pretrained checkpoint) and trains with `model_main_tf2.py`; the notebook's command uses `--num_train_steps=3500`.
- Restores a checkpoint, runs detection on a test image, then on a live webcam loop.
- Exports the model with `exporter_main_v2.py`, and includes a small Flask MJPEG webcam streaming demo.

## Tech stack

Python 3.9, TensorFlow 2.10, TensorFlow Object Detection API, OpenCV, NumPy, Matplotlib, labelImg, Flask, Jupyter.

## Getting started

```bash
git clone https://github.com/mrchikoo/Object_detection.git
cd Object_detection
python -m venv tfod && source tfod/bin/activate   # Windows: tfod\Scripts\activate
pip install tensorflow==2.10.0 opencv-python matplotlib wget flask jupyter
jupyter notebook
```

1. Run `img_coll.ipynb` to capture images, then label them with labelImg and split them into `train/` and `test/` folders under `Tensorflow/workspace/images/`.
2. Run `training.ipynb` top to bottom (protobuf compiler setup differs for Windows vs Linux; both branches are in the notebook).
3. Run the training command, then the detection cells to test on an image or your webcam.

## Repository contents

| File | Description |
|------|-------------|
| `img_coll.ipynb` | Webcam image collection and labelImg setup |
| `training.ipynb` | TFRecord generation, config, training, detection, export, Flask demo |

## Notes

- The notebooks were run on Windows with Python 3.9; paths and some shell commands (`!mkdir`, `!move`) are OS-specific, with Linux/macOS branches included.
- The dataset (images, annotations, trained checkpoints) is **not** in this repo; you need to collect and label your own.
- Notebook cells contain saved outputs from a local machine, and some cells are experimental leftovers (e.g. the training command appears interrupted in the saved output, and the export cell references `my_ssd_mobnet` rather than the `umm_last_model` name set at the top - adjust these to your own model name).
- No accuracy figures are reported in the notebooks.
