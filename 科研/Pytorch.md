



# 张量

创建一个没有初始化的5*3矩阵：

```python
torch.empty(5.3)
```

创建一个随机初始化矩阵：

```python
torch.rand(5, 3)
```

构造一个填满 0 且数据类型为 long 的矩阵：

```python
torch.zeros(5, 3, dtype=torch.long)
```

直接从数据构造张量：

```python
torch.tensor([5.5, 3])
```

根据现有的 `tensor` 建立新的 `tensor` 。

```python
#返回一个与size大小相同的用1填充的张量。
#默认情况下，返回的Tensor具有与此张量相同的torch.dtype和torch.device。
x = x.new_ones(5, 3)

# 重载 dtype! size与x相同
torch.randn_like(x, dtype=torch.float) 
```

获取张量的形状：

```python
x.size()

# 输出
torch.Size([5, 3])
```

> 注意：torch.Size 本质上还是 tuple ，所以支持 tuple 的一切操作。

# 运算

一种运算有多种语法。在下面的示例中，我们将研究加法运算。

```python
x + y
```

```python
torch.add(x, y)
```
给定一个输出张量作为参数
```python
result = torch.empty(5, 3)
torch.add(x, y, out=result)
```

原位/原地操作（in-place）

```python
y.add_(x)
```

> 注意：任何一个就地改变张量的操作后面都固定一个 `_` 。例如 `x.copy_(y)`， `x.t_（）`将更改x
>
> 但是原位操作虽然可以节省空间，但是在计算导数可能丢失历史记录，因此不推荐使用它们

```python
# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

y3 = torch.rand_like(y1)
torch.matmul(tensor, tensor.T, out=y3)


# This computes the element-wise product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z2 = tensor.mul(tensor)

z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out=z3)
```

---

也可以使用像标准的 NumPy 一样的各种索引操作：

```python
x[:, 1]
```

如果想改变形状，可以使用 `torch.view`

```python
x = torch.randn(4,4)
y = x.view(16)
z = x.view(8,-1)
print(x.size(),y.size(),z.size())

#输出
# torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
```

如果是仅包含一个元素的 `tensor`，可以使用 .item（） 来得到对应的 python 数值

```python
x = torch.randn(1)
print(x)
print(x.item())

# 输出
# tensor([0.0445])
# 0.0445479191839695
```



# 桥接NumPy

将 torch 的 Tensor 转换为 NumPy 数组

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

# 输出
# tensor([1., 1., 1., 1., 1.])
# [1. 1. 1. 1. 1.]
```

看 NumPy 细分是如何改变里面的值的：

```python
a.add_(1)
print(a)
print(b)

# 输出
# tensor([2., 2., 2., 2., 2.])
# [2. 2. 2. 2. 2.]
```
将 NumPy 数组转化为Torch张量
看改变 NumPy 分配是如何自动改变 Torch 张量的：

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a,1,out=a)
print(a)
print(b)

# 输出
# [2. 2. 2. 2. 2.]
# tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

连接tensors可以使用`torch.cat`或者`torch.stack`，用法略有不同

```python
x = torch.ones(2,2)
print(torch.cat([x,x],dim=0))
# 输出
# tensor([[1., 1.],
#        [1., 1.],
#        [1., 1.],
#        [1., 1.]])
```



**CPU上的所有张量（ CharTensor 除外）都支持与 Numpy 的相互转换。**

张量可以使用 `.to` 方法移动到任何设备（device）上：

```python
# 当GPU可用时,我们可以运行以下代码
# 我们将使用`torch.device`来将tensor移入和移出GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # 直接在GPU上创建tensor
    x = x.to(device)                       # 或者使用`.to("cuda")`方法
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to`也能在移动时改变dtype
    
# 输出
# tensor([1.0445], device='cuda:0')
# tensor([1.0445], dtype=torch.float64)
```



# Autograd自动求导

`torch.Tensor`是这个包的核心类。如果设置它的属性 `.requires_grad`为`True`，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用`.backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到`.grad`属性. 要阻止一个张量被跟踪历史，可以调用`.detach()`方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。

为了防止跟踪历史记录（和使用内存），可以将代码块包装在`with torch.no_grad():`中。在评估模型时特别有用，因为模型可能具有`requires_grad = True`的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。 还有一个类对于autograd的实现非常重要：`Function`。 `Tensor`和`Function`互相连接生成了一个非循环图，它编码了完整的计算历史。每个张量都有一个`.grad_fn`属性，它引用了一个创建了这个`Tensor`的`Function`（除非这个张量是用户手动创建的，即这个张量的`grad_fn`是`None`）。 如果需要计算导数，可以在`Tensor`上调用`.backward()`。如果`Tensor`是一个标量（即它包含一个元素的数据），则不需要为`backward()`指定任何参数，但是如果它有更多的元素，则需要指定一个`gradient`参数，它是形状匹配的张量。

```python
# 创建一个张量并设置requires_grad=True用来追踪其计算历史，如果没有指定的话，默认输入的这个标志是False。
# 可以使用.requires_grad_(true/false) 原地改变了现有张量的 requires_grad 标志
x = torch.ones(2,2,requires_grad=True)
print(x)
y = x+2
print(y)
# y是计算的结果，所以它有grad_fn属性。
print(y.grad_fn)
# 输出
# <AddBackward0 object at 0x126dfff70>
z = y*y*3
out = z.mean()
print(z)
print(out)
# 因为out是一个标量。所以让我们直接进行反向传播，out.backward()和out.backward(torch.tensor(1.))等价
out.backward()
# 输出导数d(out)/dx
print(x.grad)
# 输出
# tensor([[4.5000, 4.5000],
#        [4.5000, 4.5000]])
```

**注意**当多次调用`backward`时，参数返回的梯度也是不同的，因为PyTorch会累加梯度。如果想要正确的梯度，需要在计算前将`grad`清零。

```python
inp = torch.eye(5, requires_grad=True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

# 输出
# First call
# tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.],
#         [2., 2., 2., 2., 4.]])

# Second call
# tensor([[8., 4., 4., 4., 4.],
#         [4., 8., 4., 4., 4.],
#         [4., 4., 8., 4., 4.],
#         [4., 4., 4., 8., 4.],
#         [4., 4., 4., 4., 8.]])

# Call after zeroing gradients
# tensor([[4., 2., 2., 2., 2.],
#         [2., 4., 2., 2., 2.],
#         [2., 2., 4., 2., 2.],
#         [2., 2., 2., 4., 2.],
#         [2., 2., 2., 2., 4.]])
```

> 前面我们调用`backward()`没有使用参数，这等价于调用`backward(torch.tensor(1.0))`，这对于结果是标量的函数是有用的，比如在训练神经网络时的顺势

---

为了防止跟踪历史记录（和使用内存），可以将代码块包装在`with torch.no_grad():`中。在评估模型时特别有用，因为模型可能具有`requires_grad = True`的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。 也可以通过将代码块包装在 `with torch.no_grad():` 中，来阻止autograd跟踪设置了 `.requires_grad=True` 的张量的历史记录。

```Python
print(x.requires_grad)
print((x**2).requires_grad)

with torch.no_grad():
    print((x**2).requires_grad)

print((x**2).detach().requires_grad)
# 输出
# True
# True
# False
# False
```



# DATASETS & DATALOADERS

## 载入数据

载入[FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist)数据，包含了60000个训练数据和10000个测试数据，每个样本是28*28像素的灰度图像，对应的标签有10种类型。

```python
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# root 数据存储的路径
# train 指定是训练数据还是测试数据
# download 如果在root目录下不存在数据，则从网络上下载
# transform 和 traget_transform 指定特征和标签的转换
# ToTensor 将shape为(H,W,C)的numpy.ndarray或img转为shape为(C,H,W)的tensor，其将每一个数值归一化到[0,1]
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
```

## 遍历和可视化数据

```python
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    # squeeze的作用就是对tensor变量进行维度压缩，去除维数为1的的维度。
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

![Ankle Boot, Bag, Dress, Coat, Coat, Shirt, Dress, Shirt, T-Shirt](https://pytorch.org/tutorials/_images/sphx_glr_data_tutorial_001.png)

## 创建自定义数据集

自定义数据集一定要实现三个函数`__init__`、`__len__`、`__getitem__`，下面的这个例子中，图片存储在`img_dir`，对应的标签存储在CSV文件`annotations_file`中。

```python
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

## 准备数据参与训练

可以将数据集加载到`DataLoader`中进行遍历，每次遍历都会返回一批`train_features`和`train_labels`，如果指定`shuffle=True`，迭代到的数据都会是打乱随机的

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

# TRANSFORMS

可以使用torchvison.transforms去操纵数据数据集来使其更适合去训练，库中提供了很多常用的转换函数。

```python
import torch
from torchvision import datasets
# 使用transforms.Lambda封装其为transforms策略
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
  # scatter(dim,index,value)
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)
```

# 构建神经网络

```python
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 查看GPU是否可用，若不可用则继续使用CPU
# device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("mps")  # M1 GPU加速
print(f"Using {device} device")


# 在nn.Module类的基础上定义神经网络，使用init函数初始化，对输入数据的操作都放在forward中
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):©©
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
# 将神经网络移入device中，并且打印它的结构
model = NeuralNetwork().to(device)   # M1使用GPU加速
print('使用的计算设备是:',next(model.parameters()).device)
print(model)

# 测试
X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
```

---

```python
input_image = torch.rand(3,28,28)
print('输入数据大小：', input_image.size())

flatten = nn.Flatten()
flat_image = flatten(input_image)
print('经过Flatten层后大小：',flat_image.size())

layer1 = nn.Linear(in_features = 28*28, out_features = 20)
hidden1 = layer1(flat_image)
print('经过连接层后大小：',hidden1.size())

hidden1 = nn.ReLU()(hidden1)
print('经过激活函数后大小：',hidden1.size())

# 将各层有序放入Sequential模块中
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# 数据经过最后一个连接层后会返回logits
#  logits是[0,1]间代表模型预测每个类的概率
# dim参数指出哪个维度的值相加为1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
```

# 优化模型参数

返回模型参数，可以使用`parameters`或者`named_parameters`方法

```python
print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
```

---

这里我们先加载数据和构建模型

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
from torch.utils.data.dataloader import default_collate

device = torch.device("mps")  # M1 GPU加速
print(f"Using {device} device")

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)


# 将数据放入GPU中
train_dataloader = DataLoader(training_data, batch_size=64,
                              collate_fn=lambda x: [y.to(device) for y in default_collate(x)])
test_dataloader = DataLoader(test_data, batch_size=64,
                            collate_fn=lambda x: [y.to(device) for y in default_collate(x)])

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
```

---

超参数是可调节的参数，可以控制模型的优化进程，不同的超参数数值会影响模型训练和收敛的速率

这里定义三个超参数：

- 训练的次数Epochs
- 一次训练多少样本Batch Size
- 学习率Learning Rate

```python
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

---

每个epoch包含了两个部分

- 训练Loop，在训练集上迭代并尝试收敛达到最佳
- 验证/测试Loop，验证模型表现是否得到提升

---

损失函数计算了模型结果和目标结果不同的程度，训练时需要降低损失函数的数值

常见的损失函数有`nn.MSELoss`（平均平方差，回归任务），`nn.NLLLoss`（Negative Log Likelihood，分类任务），`nn.CrossEntropyLoss`结合了`nn.LogSoftmax`和`nn.NLLLoss`

我们将模型输出logits传入`nn.CrossEntropyLoss`，它会normalize the logits 并计算损失

```python
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss(
```

---

这里定义一个优化器

```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

在训练loop中，优化发生在三步：

- 调用`optimizer.zero_grad()`来重置模型参数的梯度
- 调用`loss.backward()`反向传播预测损失
- 当拿到了参数梯度，调用`optimizer.step()`去调整参数

---

我们定义优化代码在`train_loop`函数中，在`test_loop`函数中测试模型的表现

```python
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

最后我们初始化损失函数和优化器，将它们传入`train_loop`和`test_loop`中

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

# 保存和加载模型

导入库

```python
import torch
import torchvision.models as models
```

模型在内部的状态字典`state_dict`中存储学习参数，参数可以使用函数`torch.save`来保存

```python
model = models.vgg16(pretrained=True)
torch.save(model.state_dict(), 'model_weights.pth')
```

加载模型参数，需要先创建一个相同的模型结构，然后使用函数`load_state_dict()`来加载参数

```python
model = models.vgg16() # we do not specify pretrained=True, i.e. do not load default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()
```

> 确保在使用模型前调用`model.eval()`函数，来让dropout和batch normalization层进入测试模式（不启用），不然在测试中会改变权值，影响结果
>
> 与之相对应的是`model.train()`

如果需要一起保存模型结构和参数，可以传入`model`而不是`model.state_dict()`去保持整个模型

```python
torch.save(model, 'model.pth')
```

像这样加载模型

```python
model = torch.load('model.pth')
```

> 这个方法在序列化模型时使用了Python的pickle模块，因此在加载模型时依赖可用的实际类定义

