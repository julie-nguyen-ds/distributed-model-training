# distributed-model-training
This repository is intended to help benchmark the performance of various clusters configurations for distributed and parallelized Computer Vision model transfer training. 

![Horovod and Pytorch Logo](/images/horovod_pytorch_logo.png)

The main notebook uses transfer learning to tune the pre-trained [Faster-RCNN](https://pytorch.org/vision/main/models/faster_rcnn.html) Pytorch model on face mask detection task on single node vs multiple nodes using Horovod. The results are tracked using mlflow experiment tracking for easier benchmarking.

![Mlflow tracking](/images/mlflow_experiments.png)

## Use case
In this project, we use the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) from Kaggle as our example use-case to detect whether individuals on picture are wearing a mask or not, and detect the mask on the picture. 

![Mask Detection example output](/images/face_mask_detection.png)

## Technical stacks
- Pytorch: Used to download pretrained [Faster-RCNN](https://pytorch.org/vision/main/models/faster_rcnn.html) and fine-tune/train the model to the usecase.
- Horovod: Used to distribute training across cluster workers, can be CPU or GPU clusters. 
- Mlflow tracking: To keep track of training performances between distributed / non-distributed training, and clusters configurations.


## How to use this repository
This demo is hosted in [e2-demo-field-eng workspace](e2-demo-field-eng.cloud.databricks.com/). 

If the dataset is not found, follow step 1 to 3. Else, skip to step 4 directly.

1. Download the [Face Mask Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) from Kaggle.
2. Import the dataset to your Databricks workspace in DBFS.
3. Edit the path in the first cell of the [main.py notebook](https://github.com/julie-nguyen-ds/distributed-model-training/blob/main/main.py) to your DBFS dataset location. 
4. Run and follow instruction on the main notebook using your desired cluster configuration.
5. The experiments and training will be logged to mlflow with your cluster configuration. eg: driver/worker node type, training time, number of processors used.
