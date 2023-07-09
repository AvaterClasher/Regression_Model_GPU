# Regression_Model_GPU

<div class="cell markdown" id="IX_WDhvpZQKg">

# **Device Agnostic Regression Model using Pytorch**

</div>

<div class="cell markdown" id="6FPGLoq7ZvKs">

## Importing Modules

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:36}"
id="vYkYiaybGU7I" outputId="693ed485-31f5-4262-c5d8-b0af60e1c82c">

``` python
# Import PyTorch and matplotlib
import torch
from torch import nn # nn contains all of PyTorch's building blocks for neural networks
import matplotlib.pyplot as plt

# Check PyTorch version
torch.__version__
```

<div class="output execute_result" execution_count="30">

``` json
{"type":"string"}
```

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="fPbUGhFBGj93" outputId="772d36f9-05cf-4540-9ed3-7ecc6cef7de5">

``` python
# Setup device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

<div class="output stream stdout">

    Using device: cuda

</div>

</div>

<div class="cell markdown" id="IodnD3yzZ1Sy">

## Creating Parameters

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="Paodt4YLGmoo" outputId="37fc71dd-7203-4b89-d326-70a59e973b88">

``` python
# Create weight and bias
weight = 0.7
bias = 0.3

# Create range values
start = 0
end = 1
step = 0.02

# Create X and y (features and labels)
X = torch.arange(start, end, step).unsqueeze(dim=1) # without unsqueeze, errors will happen later on (shapes within linear layers)
y = weight * X + bias
X[:10], y[:10]
```

<div class="output execute_result" execution_count="32">

    (tensor([[0.0000],
             [0.0200],
             [0.0400],
             [0.0600],
             [0.0800],
             [0.1000],
             [0.1200],
             [0.1400],
             [0.1600],
             [0.1800]]),
     tensor([[0.3000],
             [0.3140],
             [0.3280],
             [0.3420],
             [0.3560],
             [0.3700],
             [0.3840],
             [0.3980],
             [0.4120],
             [0.4260]]))

</div>

</div>

<div class="cell markdown" id="eJhWH8_OaHDi">

## Creating Training and Testing Datasets

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="vaOpB4qMHdly" outputId="acb3beab-b00f-4a66-b522-8e1f6a486a8c">

``` python
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split],y[:train_split]
X_test, y_test = X[train_split:],y[train_split:]

len(X_train),len(y_train),len(X_test),len(y_test)
```

<div class="output execute_result" execution_count="33">

    (40, 40, 10, 10)

</div>

</div>

<div class="cell markdown" id="D8ibuFvXaLxC">

## Plotting Functions

</div>

<div class="cell code" id="nDXR7aSsHEx7">

``` python
## Plotting Function ##
def plot_predictions(train_data=X_train,train_labels=y_train,test_data=X_test,test_labels=y_test,predictions=None):
   plt.figure(figsize=(10,7))
   # pLot training data in blue
   plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
   # plot testing data in green
   plt.scatter(test_data,test_labels,c="g",s=4,label="Testing data")
   # Are there predictions ?
   if predictions is not None:
       plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")
   plt.legend(prop={"size":14});
```

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:472}"
id="pExzzjqQGpJD" outputId="8d0a408c-719d-4c94-83ca-c3222da93983">

``` python
# Note: If you've reset your runtime, this function won't work,
# you'll have to rerun the cell above where it's instantiated.
plot_predictions(X_train, y_train, X_test, y_test)
```

<div class="output display_data">

![](7f448feaa430fa2daa35c9613fed06c7de4504c2.png)

</div>

</div>

<div class="cell markdown" id="122mgI2raQbS">

## Regression Model

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="0wnhUnUjG8xy" outputId="41e8c403-3e1b-4a7a-fd8b-45a19a40bcd0">

``` python
# Subclass nn.Module to make our model
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() for creating the model parameters
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)

    # Define the forward computation (input data x flows through nn.Linear())
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_layer(x)

# Set the manual seed when creating the model (this isn't always need but is used for demonstrative purposes, try commenting it out and seeing what happens)
torch.manual_seed(42)
model_1 = LinearRegressionModelV2()
model_1, model_1.state_dict()
```

<div class="output execute_result" execution_count="36">

    (LinearRegressionModelV2(
       (linear_layer): Linear(in_features=1, out_features=1, bias=True)
     ),
     OrderedDict([('linear_layer.weight', tensor([[0.7645]])),
                  ('linear_layer.bias', tensor([0.8300]))]))

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="BhlKpWSGIAjF" outputId="0a6a05f8-ad70-42e6-d835-f2e97e1c81b1">

``` python
# Check model device
next(model_1.parameters()).device
```

<div class="output execute_result" execution_count="37">

    device(type='cpu')

</div>

</div>

<div class="cell markdown" id="LvZzUuAmaa_w">

## Transfering our model to the **GPU** `(if it is free)`

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="O4VfS7LMICHl" outputId="99258000-7b6b-40b2-f132-d109b0b5dfce">

``` python
# Set model to GPU if it's availalble, otherwise it'll default to CPU
model_1.to(device) # the device variable was set above to be "cuda" if available or "cpu" if not
next(model_1.parameters()).device
```

<div class="output execute_result" execution_count="38">

    device(type='cuda', index=0)

</div>

</div>

<div class="cell markdown" id="iSFOawgtapyE">

## Setting up the Loss Function `(L1Loss)` and The Optimiser `(SGD)`

</div>

<div class="cell code" id="GC81uhoxJsIG">

``` python
# Create loss function
loss_fn = nn.L1Loss()

# Create optimizer
optimizer = torch.optim.SGD(params=model_1.parameters(), # optimize newly created model's parameters
                            lr=0.01)
```

</div>

<div class="cell markdown" id="RxeWpZKZa5vq">

## **Training and Testing Loop**

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="JDOHzX8lec09" outputId="03b1747d-1ff5-43fc-b5d0-7253c903b453">

``` python
torch.manual_seed(42)

# Set the number of epochs
epochs = 100000

# Put data on the available device
# Without this, error will happen (not all model/data on device)
X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(epochs):
    ### Training
    model_1.train() # train mode is on by default after construction

    # 1. Forward pass
    y_pred = model_1(X_train)

    # 2. Calculate loss
    loss = loss_fn(y_pred, y_train)

    # 3. Zero grad optimizer
    optimizer.zero_grad()

    # 4. Loss backward
    loss.backward()

    # 5. Step the optimizer
    optimizer.step()

    ### Testing
    model_1.eval() # put the model in evaluation mode for testing (inference)
    # 1. Forward pass
    with torch.inference_mode():
        test_pred = model_1(X_test)

        # 2. Calculate the loss
        test_loss = loss_fn(test_pred, y_test)

    if epoch % 50000 == 0:
        print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")
```

<div class="output stream stdout">

    Epoch: 0 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882
    Epoch: 50000 | Train loss: 0.0012645035749301314 | Test loss: 0.013801801018416882

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="4WT2JNh2VXqC" outputId="818121db-fa30-4613-9d56-ca6c83fe1abf">

``` python
# Find our model's learned parameters
from pprint import pprint # pprint = pretty print, see: https://docs.python.org/3/library/pprint.html
print("The model learned the following values for weights and bias:")
pprint(model_1.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")
```

<div class="output stream stdout">

    The model learned the following values for weights and bias:
    OrderedDict([('linear_layer.weight', tensor([[0.6968]], device='cuda:0')),
                 ('linear_layer.bias', tensor([0.3025], device='cuda:0'))])

    And the original values for weights and bias are:
    weights: 0.7, bias: 0.3

</div>

</div>

<div class="cell markdown" id="jV_-AKoQbDLz">

## Printing the Testing Loops Values

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="6F22wlqLVbsN" outputId="0d7f5620-bdb7-49ba-f61c-60e40f42525d">

``` python
# Turn model into evaluation mode
model_1.eval()

# Make predictions on the test data
with torch.inference_mode():
    y_preds = model_1(X_test)
y_preds
```

<div class="output execute_result" execution_count="42">

    tensor([[0.8600],
            [0.8739],
            [0.8878],
            [0.9018],
            [0.9157],
            [0.9296],
            [0.9436],
            [0.9575],
            [0.9714],
            [0.9854]], device='cuda:0')

</div>

</div>

<div class="cell markdown" id="kANfICGkbJgg">

## Plotting the testing loops values

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:472}"
id="FV1rjWvMVcmi" outputId="d924a3a0-b395-4da6-c25f-d1b25809846d">

``` python
# plot_predictions(predictions=y_preds) # -> won't work... data not on CPU

# Put data on the CPU and plot it
plot_predictions(predictions=y_preds.cpu())
```

<div class="output display_data">

![](b616a81cbeb0124f34cfb820345dc604b000bc5e.png)

</div>

</div>

<div class="cell markdown" id="n2GKrQZGbPJX">

## Saving the Models data

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="W88nJieGVf7L" outputId="decbaed4-3c12-4a41-9d46-fbe8b14f50b6">

``` python
from pathlib import Path

# 1. Create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# 2. Create model save path
MODEL_NAME = "Regression_Model_State_Dict.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# 3. Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_1.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)
```

<div class="output stream stdout">

    Saving model to: models/Regression_Model_State_Dict.pth

</div>

</div>

<div class="cell markdown" id="yumAnNj6bVFy">

## Loading the models data

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="PAbNrzv3Vh_K" outputId="21a4bea7-8917-4f12-d761-a63ada5f1711">

``` python
# Instantiate a fresh instance of LinearRegressionModelV2
loaded_model_1 = LinearRegressionModelV2()

# Load model state dict
loaded_model_1.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Put model to target device (if your data is on GPU, model will have to be on GPU to make predictions)
loaded_model_1.to(device)

print(f"Loaded model:\n{loaded_model_1}")
print(f"Model on device:\n{next(loaded_model_1.parameters()).device}")
```

<div class="output stream stdout">

    Loaded model:
    LinearRegressionModelV2(
      (linear_layer): Linear(in_features=1, out_features=1, bias=True)
    )
    Model on device:
    cuda:0

</div>

</div>

<div class="cell code"
colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}"
id="fb3AjbvEVjpN" outputId="ef221f22-c38b-4838-8d5e-994ae9a9bfaa">

``` python
# Evaluate loaded model
loaded_model_1.eval()
with torch.inference_mode():
    loaded_model_1_preds = loaded_model_1(X_test)
y_preds == loaded_model_1_preds
```

<div class="output execute_result" execution_count="46">

    tensor([[True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True],
            [True]], device='cuda:0')

</div>

</div>
