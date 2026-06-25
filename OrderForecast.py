# Databricks notebook source
# MAGIC %load_ext autoreload
# MAGIC %autoreload 2
# MAGIC # Enables autoreload; learn more at https://docs.databricks.com/en/files/workspace-modules.html#autoreload-for-python-modules
# MAGIC # To disable autoreload; run %autoreload 0

# COMMAND ----------

# DBTITLE 1,Install PyTorch
# MAGIC %pip install torch

# COMMAND ----------

# DBTITLE 1,Cell 2
import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
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
from OrderForecastDS import OFDataset

import OrderForecastFNN
importlib.reload(OrderForecastFNN)
from OrderForecastFNN import OFANN

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)


        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


pdc = 3131
sDate = '2022-01-01'
eDate = '2022-04-30'
trainData = OFDataset(spark, pdc, sDate, eDate)


batch_size = 16
# Create data loaders.
train_dataloader = DataLoader(trainData, batch_size=batch_size, shuffle=True)

# Display train data and labels.
train_features, train_labels = next(iter(train_dataloader))
print(f"Dataset size: {trainData.len}")
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
#print(f"Features: {train_features}")
#print(f"Labels: {train_labels}")


# Create model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = OFANN(trainData.predictorLenD, trainData.responseLenD).to(device)
print(model)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

epochs = 1000
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
print("Done!")