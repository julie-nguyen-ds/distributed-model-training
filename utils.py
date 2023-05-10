# Databricks notebook source
# MAGIC 
%pip install beautifulsoup4 lxml

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

### Configure Databricks Token in notebook
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
dbutils.fs.put("file:///root/.databrickscfg",
               "[DEFAULT]\nhost=https://community.cloud.databricks.com\ntoken = " + token, overwrite=True)

# COMMAND ----------

# MAGIC %md

# COMMAND ----------

### Create method to show images
from PIL import Image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

colour_mapping = {
    0: "red",
    1: "aqua",
    2: "violet"
}

label_mapping = {
    0: "No Mask",
    1: "With Mask",
    2: "Incorrect Mask"
}

def show_image(image_path, preds, threshold=0.5, dpi=50):
    font_size = 40
    line_width = 5

    image = Image.open(image_path)

    img_width, img_height = image.size
    draw = ImageDraw.Draw(image)

    #     # calculate and display bounding boxes for each detected custom label
    image_level_label_height = 0

    for box, label, score in zip(preds['boxes'], preds['labels'], preds['scores']):

        if score > threshold:
            confidence = int(round(score * 100, 0))
            label_text = f"{label_mapping[label]}:{confidence}%"
            text_width, text_height = draw.textsize(label_text)

            draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline=colour_mapping[label])
            draw.rectangle([(box[0] + line_width, box[1] + line_width), (box[0] + text_width, box[1] + text_height)],
                           fill="black")
            draw.text((box[0] + line_width, box[1] + line_width), label_text, fill=colour_mapping[label])

    plt.figure(figsize=(img_width / dpi, img_height / dpi))
    plt.imshow(image, interpolation='nearest', aspect='auto')
    
def show_image_from_tensor(img_tensor, annotation):
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box.detach().cpu()

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()

# COMMAND ----------

### Load method to draw bounding boxes
import numpy as np
import pandas as pd
import os

from bs4 import BeautifulSoup

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms, datasets, models
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import PIL
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.patches as patches


# COMMAND ----------

def generate_box(obj):
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0


def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id

        return target


# COMMAND ----------

def collate_fn(batch):
    return tuple(zip(*batch))

# COMMAND ----------

### Method to load PyTorch pre trained FasterRCNN model
def get_object_detect_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    print("in_features:", in_features)
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# COMMAND ----------



# COMMAND ----------


