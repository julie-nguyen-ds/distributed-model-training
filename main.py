# Databricks notebook source
# MAGIC %md
# MAGIC # Distributed Deep Learning training using PyTorch with HorovodRunner for Mask Detection
# MAGIC 
# MAGIC This notebook illustrates the use of HorovodRunner for distributed training using PyTorch.
# MAGIC It first shows how to train a model on a single node, and then shows how to adapt the code using HorovodRunner for distributed training.
# MAGIC The notebook runs on both CPU and GPU clusters.
# MAGIC 
# MAGIC ## Setup Requirements
# MAGIC Databricks Runtime 7.6 ML or above (choose either CPU or GPU clusters). Torch and its related packages and Horovod are preinstalled with this runtime.
# MAGIC HorovodRunner is designed to improve model training performance on clusters with multiple workers, but multiple workers are not required to run this notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load utility functions
# MAGIC For image display, bounding box and pre-trained model loading.

# COMMAND ----------

# MAGIC %run ./utils

# COMMAND ----------

# MAGIC %md ### Set up checkpoint location
# MAGIC The next cell creates a directory for saved checkpoint models. Databricks recommends saving training data under `dbfs:/ml`, which maps to `file:/dbfs/ml` on driver and worker nodes.

# COMMAND ----------

# Checkpoint directory
PYTORCH_DIR = '/dbfs/ml/horovod_pytorch'

# Dataset: CHANGE PATH TO YOUR DATASET DBFS LOCATION IF NEEDED
IMAGE_PATH = "/dbfs/FileStore/changshi.lim/face_mask_dataset/images"
ANNOTATION_PATH = "/dbfs/FileStore/changshi.lim/face_mask_dataset/annotations"

# COMMAND ----------

# MAGIC %md ### Configure model parameters & cluster info tracking

# COMMAND ----------

# Specify training parameters
batch_size = 4
num_epochs = 1
momentum = 0.9
log_interval = 100
weight_decay = 0.0005
learning_rate = 0.005
driver_node_type = spark.conf.get("spark.databricks.clusterUsageTags.driverNodeType")
worker_node_type = spark.conf.get("spark.databricks.workerNodeTypeId")
number_workers = spark.conf.get("spark.databricks.clusterUsageTags.clusterWorkers")
spark_version = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")


def log_training_metadata_to_mlflow(batch_size=batch_size, num_epochs=num_epochs,
                                    spark_version=spark_version, driver_node_type=driver_node_type,
                                    worker_node_type=worker_node_type, number_workers=number_workers, 
                                    is_distributed_training=False, number_distributed_processors=1) -> None:
    mlflow.log_param("batch_size", batch_size)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.set_tag("spark_version", spark_version)

    mlflow.set_tag("driver_node_type", driver_node_type)
    mlflow.set_tag("worker_node_type", worker_node_type)
    mlflow.set_tag("number_workers", number_workers)

    mlflow.set_tag("distributed_training", is_distributed_training)
    mlflow.set_tag("number_processors_used", number_distributed_processors)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mlflow.set_tag("device", device)


# COMMAND ----------

# MAGIC %md ### Methods for saving and loading model checkpoints

# COMMAND ----------

def save_checkpoint(log_dir, model, optimizer, epoch):
    filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filepath)


def load_checkpoint(log_dir, epoch=num_epochs):
    filepath = log_dir + '/checkpoint-{epoch}.pth.tar'.format(epoch=epoch)
    return torch.load(filepath)


def create_log_dir():
    log_dir = os.path.join(PYTORCH_DIR, str(time()), 'MaskDetectionDemo')
    os.makedirs(log_dir)
    return log_dir

# COMMAND ----------

# MAGIC %md ### Load Pretrained Model (FasterRCNN)

# COMMAND ----------

num_class = 3
rcnn_model = get_object_detect_model(num_class)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Mask Dataset

# COMMAND ----------

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class MaskDataset(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(IMAGE_PATH)))
        self.targets = list(sorted(os.listdir(ANNOTATION_PATH))) 
        
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        file_image = 'maksssksksss' + str(idx) + '.png'
        file_label = 'maksssksksss' + str(idx) + '.xml'
        img_path = os.path.join(IMAGE_PATH, file_image)
        label_path = os.path.join(ANNOTATION_PATH, file_label)
        img = Image.open(img_path).convert("RGB")

        # Generate Label
        target = generate_target(idx, label_path)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

# Import the pictures and bounding boxes
data_transform = transforms.Compose([transforms.ToTensor()])
dataset = MaskDataset(data_transform)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create method to train model for one epoch

# COMMAND ----------

def train_one_epoch(model, device, data_loader, optimizer, epoch, activate_mlflow=True):
    model.train()

    # 3.0 Compute loss and backpropagate
    for batch_idx, (imgs, annotations) in enumerate(data_loader):
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
        loss_dict = model([imgs[0]], [annotations[0]])
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # 4.0 Print progress
        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(imgs),
                    len(data_loader.dataset),
                    100.0 * batch_idx / len(data_loader),
                    losses)
            )
        
        # 5.0 Log progress to Tensorboard and MLflow
        if activate_mlflow:
            mlflow.log_metric("train loss", losses.item(), epoch * len(data_loader) + batch_idx)
            
    print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(epoch, batch_idx * len(imgs), len(data_loader.dataset), 100.0 * batch_idx / len(data_loader), losses))

# COMMAND ----------

# MAGIC %md ## Prepare single-node model training with PyTorch
# MAGIC Run single-node training with PyTorch and record training parameters.

# COMMAND ----------

import pickle
import tempfile
import os
import mlflow
import torch.optim as optim
from torchvision import datasets, transforms
from time import time

output_dir = tempfile.mkdtemp()
single_node_log_dir = create_log_dir()


def train(train_dataset, learning_rate):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn)

    model = rcnn_model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    with mlflow.start_run() as run:
        log_training_metadata_to_mlflow()

        for epoch in range(1, num_epochs + 1):
            train_one_epoch(model, device, data_loader, optimizer, epoch)
            save_checkpoint(single_node_log_dir, model, optimizer, epoch)

        mlflow.log_artifacts(output_dir, artifact_path="events")
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)
        run_id = run.info.run_id
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC Run the `train` function you just created to train a model on the driver node.

# COMMAND ----------

model = train(train_dataset=train_dataset, learning_rate=learning_rate)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Migrate to HorovodRunner for Distributed Training
# MAGIC 
# MAGIC HorovodRunner takes a Python method that contains deep learning training code with Horovod hooks. HorovodRunner pickles the method on the driver and distributes it to Spark workers. A Horovod MPI job is embedded as a Spark job using barrier execution mode.

# COMMAND ----------

import horovod.torch as hvd
from sparkdl import HorovodRunner

# COMMAND ----------

from torch.utils.data.distributed import DistributedSampler
hvd_log_dir = create_log_dir()

def train_hvd(learning_rate, dataset):
    # Initialize Horovod
    hvd.init()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)

    # Pin GPU to local rank
    if device.type == 'cuda':
        torch.cuda.set_device(hvd.local_rank())

    # Configure the sampler so that each worker gets a distinct sample of the input dataset
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    print("size ", hvd.size(), " and rank", hvd.rank())

    # Use train_sampler to load a different sample of data on each worker
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                               collate_fn=collate_fn)

    model = rcnn_model.to(device)

    # The effective batch size in synchronous distributed training is scaled by the number of workers
    # Increase learning_rate to compensate for the increased batch size
    optimizer = optim.SGD(model.parameters(), lr=learning_rate * hvd.size(), momentum=momentum)

    # Wrap the local optimizer with hvd.DistributedOptimizer so that Horovod handles the distributed optimization
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())

    # Broadcast initial parameters so all workers start with the same parameters
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    for epoch in range(1, num_epochs + 1):
        train_one_epoch(model, device, train_loader, optimizer, epoch, activate_mlflow=False)
        # Save checkpoints only on worker 0 to prevent conflicts between workers
        if hvd.rank() == 0:
            save_checkpoint(hvd_log_dir, model, optimizer, epoch)

    return model


# COMMAND ----------

# MAGIC 
# MAGIC %md
# MAGIC Now that you have defined a training function with Horovod,  you can use HorovodRunner to distribute the work of training the model.
# MAGIC 
# MAGIC The HorovodRunner parameter `np` sets the number of processes. This example uses a cluster with two workers, each with a single GPU, so set `np=2`. (If you use `np=-1`, HorovodRunner trains using a single process on the driver node.)
# MAGIC 
# MAGIC If you are using a CPU cluster, the maximum number of processes is the total number of cores of your workers. eg: 3 workers nodes with 4CPUs core could set `np=12`.

# COMMAND ----------

number_processors = 2
hr = HorovodRunner(np=number_processors, driver_log_verbosity='all')

with mlflow.start_run() as run:
    log_training_metadata_to_mlflow(is_distributed_training=True, number_distributed_processors=number_processors)
    new_model = hr.run(train_hvd, learning_rate=learning_rate, dataset=train_dataset)
    mlflow.pytorch.log_model(new_model, artifact_path="pytorch-model", pickle_module=pickle)

# COMMAND ----------

# MAGIC %md 
# MAGIC Under the hood, HorovodRunner takes a Python method that contains deep learning training code with Horovod hooks. HorovodRunner pickles the method on the driver and distributes it to Spark workers. A Horovod MPI job is embedded as a Spark job using the barrier execution mode. The first executor collects the IP addresses of all task executors using BarrierTaskContext and triggers a Horovod job using `mpirun`. Each Python MPI process loads the pickled user program, deserializes it, and runs it.
# MAGIC 
# MAGIC For more information, see [HorovodRunner API documentation](https://databricks.github.io/spark-deep-learning/#api-documentation). 
