# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Install PyTorch
# MAGIC %pip install torch
# MAGIC %restart_python

# COMMAND ----------

# DBTITLE 1,Cell 2
import torch
from datetime import timedelta

from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torchsummary import summary

import os
import torch.distributed as dist
from serverless_gpu import distributed

import time

import sys
import importlib
import importlib.util

# Manually load OrderForecastFNN module
#spec = importlib.util.spec_from_file_location("OrderForecastFNN", "/Workspace/Users/t0304lc@stellantis.com/OrderForecastFNN.py")
#OrderForecastFNN_module = importlib.util.module_from_spec(spec)
#sys.modules['OrderForecastFNN'] = OrderForecastFNN_module
#spec.loader.exec_module(OrderForecastFNN_module)

import OrderForecastDS
importlib.reload(OrderForecastDS)
from OrderForecastDS import OFDataset, mean_std_denorm

import OrderForecastFNN
importlib.reload(OrderForecastFNN)

from LrReLULayers import LrReLU
from cosSimAttention import cosPeTransformer, cosPcTransformer
from channelMult import ChannelReplicate, PerChannelLinear, channelsLinear
from OrderForecastFNN import OFANN, OFCANN
from OrderForecastTNN import OFTNN, OFTTNN, OFTTCNN, OFTTCNN2


#Parallelize training
def setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

@distributed(gpus=8, gpu_type="H100")
def p_train():
    import sys
    sys.path.insert(0, '/Workspace/Users/t0304lc@stellantis.com')

    import mlflow
    import torch.optim as optim
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
    from OrderForecastTNN import OFTTCNN2

    # Bind this process to its own GPU before training.
#    local_rank = int(os.environ["LOCAL_RANK"])
#    torch.cuda.set_device(local_rank)
#    device = torch.device(f"cuda:{local_rank}")
#    dist.init_process_group("nccl")

    # 1. Set up multi-GPU environment
    setup()
    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")

    # 3. Use pre-extracted tensor data (spark unavailable in subprocess).
    batch_size = 20
    dataset = TensorDataset(_train_Xn, _train_Yn, _train_mean, _train_std)
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, shuffle=False)

    # 4. Define the training loop.
    # 2. Apply the Torch distributed data parallel (DDP) library for data-parellel training.
    model = OFTTCNN2(_predictor_len, _response_len_d, 60, F.tanh, F.tanh).to(device)
    model = DDP(model, device_ids=[device], find_unused_parameters=True)

    loss_fn = nn.MSELoss()
    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

    # ... build the model and data on `device`, then run your training loop ...
    epochs = 2000
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        sampler.set_epoch(t)

        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y, _, _) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Log loss to MLflow metric
            #mlflow.log_metric("loss", loss.item(), step=batch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            optimizer.zero_grad()
        
            if batch % 200 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    #for epoch in range(num_epochs):
        #sampler.set_epoch(epoch)
        #model.train()
        #total_loss = 0.0
        #for step, (xb, yb) in enumerate(dataloader):
        #    xb, yb = xb.to(device), yb.to(device)
        #    optimizer.zero_grad()
        #    loss = loss_fn(model(xb), yb)
            # Log loss to MLflow metric
        #    mlflow.log_metric("loss", loss.item(), step=step)

        #    loss.backward()
        #    optimizer.step()
        #    total_loss += loss.item() * xb.size(0)

        #mlflow.log_metric("total_loss", total_loss)
        #print(f"Total loss for epoch {epoch}: {total_loss}")


#    dist.destroy_process_group()

    # Save trained weights from rank 0
    if int(os.environ.get("RANK", 0)) == 0:
        torch.save(model.module.state_dict(), '/tmp/p_train_model.pt')

    cleanup()

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y, _, _) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        if batch % 200 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y, _, _ in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            #correct += (pred.reshape(1,-1).round() == y.round()).type(torch.float).sum().item()

    test_loss /= num_batches
    #correct /= size
    #print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

def mape_loss(y_pred, y_true):
    epsilon = 1e-8  # To avoid division by zero
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon)))

#DEBUG
def experiments():
    # Create dummy input: 
    batch_size=2 
    channels=3
    features=4
    out=5
    input_tensor = torch.rand(batch_size, features)
    #input_tensor = torch.tensor([[1,0,1]], dtype=torch.float32)
    print(f"Input tensor: {input_tensor.size()}, {input_tensor}")
    
    # Create instance of custom layer
    #layer = nn.Flatten(start_dim=1)
    layer = ChannelReplicate(channels)
    layer2 = channelsLinear(channels, features, out)
    print(f"W tensor: {layer2.W.size()}, {layer2.W}, W0: {layer2.W0.size()}{layer2.W0}")
    #layer = LrReLU(features)
    #layer = cosPeTransformer(features)
    #layer = cosPcTransformer(features)
    
    # Pass input through layer
    output_tensor = layer(input_tensor)
    print(f"Output tensor: {output_tensor.size()}, {output_tensor}")

    #W = torch.rand(batch_size, channels, features, out)
    #W0 = torch.ones(batch_size, channels, out)
    #Y = torch.matmul(input_tensor.unsqueeze(-2), W).squeeze(-2)
    #Y = torch.matmul(input_tensor, W)
    #Z = Y + W0

    output_tensor2 = layer2(output_tensor)
    print(f"Output tensor: {output_tensor2.size()}, {output_tensor2}")


#experiments()

pdc = 3131
sDate = '2022-01-01'
eDate = '2025-12-31'
trainData = OFDataset(spark, pdc, sDate, eDate)

tsDate = '2026-01-01'
teDate = '2026-09-30'
testData = OFDataset(spark, pdc, tsDate, teDate)


batch_size = 20
# Create data loaders.
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(testData, batch_size=batch_size, shuffle=False)

# Display train data and labels.
#train_features, train_labels = next(iter(train_dataloader))
#print(f"Dataset size: {trainData.len}")
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#print(f"Features: {train_features}")
#print(f"Labels: {train_labels}")


# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#model = OFANN(trainData.predictorLen, trainData.responseLenD).to(device)
#model = OFCANN(trainData.predictorLen, trainData.responseLenD).to(device)

#model = OFTTNN(trainData.predictorLen, trainData.responseLenD, 60).to(device)#, F.tanh, F.tanh).to(device)
#model = OFTTCNN(trainData.predictorLen, trainData.responseLenD, 60).to(device)#, F.tanh, F.tanh).to(device)
model = OFTTCNN2(trainData.predictorLen, trainData.responseLenD, 60, F.tanh, F.tanh).to(device)#).to(device)
print(model)
#summary(model, (1, 28, 28))

loss_fn = nn.MSELoss()
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

start_t = time.time()
epochs = 2000

# Pre-extract tensor data for distributed training (spark unavailable in subprocess)
_train_Xn = trainData.Xn.T
_train_Yn = trainData.Yn.T
_train_mean = trainData.mean.T
_train_std = trainData.std.T
_predictor_len = trainData.predictorLen
_response_len_d = trainData.responseLenD

#Distributed
p_train.distributed()

# Load trained weights back into the main-process model
model.load_state_dict(torch.load('/tmp/p_train_model.pt', map_location=device, weights_only=True))
print('Distributed model weights loaded into main-process model.')

#Monolyth
#for t in range(epochs):
#    print(f"Epoch {t+1}\n-------------------------------")
#    train(train_dataloader, model, loss_fn, optimizer)

end_t = time.time()
print(f"Done! Execution time: {end_t - start_t:.6f} seconds")

test(test_dataloader, model, loss_fn)



# COMMAND ----------

import torch
from datetime import timedelta

from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
#from torchsummary import summary
import time

test_loss = 0
with torch.no_grad():
  size = len(test_dataloader.dataset)
  num_batches = len(test_dataloader)
  batch_size = test_dataloader.batch_size
  model.eval()

  #for X, Y, mean, std in testData:
  for X, Y, mean, std in test_dataloader:
    X, Y = X.to(device), Y.to(device)
    mean, std = mean.to(device), std.to(device)
    Yh = model(X)
    Yhd = mean_std_denorm(Yh, mean, std)
    Yd = mean_std_denorm(Y, mean, std)

    nBatch, _ = X.size()
  
    for i in range(0, nBatch):
      Yhd1 = Yhd[i, :] 
      Yd1 = Yd[i, :]
      mean1 = mean[i]
      std1 = std[i]
      test_loss_step = mape_loss(Yhd1, Yd1)
      test_loss += test_loss_step
      print(f"Predicted: {Yhd1}, Actual: {Yd1}, MAPE: {test_loss_step:.6f}, Mean: {mean1}, Std: {std1}")

test_loss /= len(testData)
print(f"Test MAPE Error: {test_loss:.6f}")