# 线性模型

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007095009115.png" alt="image-20221007095009115" style="zoom:67%;" />

w为 𝐷 维的权重向量，𝑏 为偏置。

在分类问题中，线性的𝑓(𝒙; 𝒘) 的值域为实数，因此无法直接用 𝑓(𝒙; 𝒘) 来进行预测，需要引入一个非线性的决策函数𝑔(⋅) 来预测输出目标

![image-20221007095433970](/Users/eumenides/Library/Application Support/typora-user-images/image-20221007095433970.png)

## 线性判别函数和决策边界

### 二分类

二分类问题的类别标签 𝑦 只有两种取值，通常可以设为 {+1, −1} 或 {0, 1}。

将**𝑓(𝒙; 𝒘) = 0**的所有点组成一个分割超平面也叫决策边界或决策平面。决策边界将特征空间一分为二，划分成两个区域，每个区域对应一个类别。

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007100122308.png" alt="image-20221007100122308" style="zoom: 33%;" />

训练集的每个样本都满足

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007100226625.png" alt="image-20221007100226625" style="zoom: 50%;" />

这样也称该训练集是线性可分的。

### 多分类

多分类指分类的类别数大于2，一般需要多个线性判别函数，设计这些判别函数有很多种方式：

1. “一对其余”，把多分类问题转换为 𝐶 个“一对其余”的二分类问题。

2. “一对一”，把多分类问题转换为 𝐶(𝐶 − 1)/2 个“一对一”的二分类问题。

3. “argmax”，共需要 𝐶 个判别函数，对于样本 𝒙，如果存在一个类别 𝑐，相对于所有的其他类别 𝑐(̃𝑐 ̃ ≠ 𝑐) 有 𝑓 𝑐 (𝒙; 𝒘 𝑐 ) > 𝑓 𝑐̃ (𝒙, 𝒘 𝑐̃ )，那么 𝒙 属于类别 𝑐。

   <img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007101036975.png" alt="image-20221007101036975" style="zoom: 67%;" />

“一对其余”方式和“一对一”方式都存在一个缺陷：特征空间中会存在一 些难以确定类别的区域，而“argmax”方式很好地解决了这个问题．

![image-20221007101230000](/Users/eumenides/Library/Application Support/typora-user-images/image-20221007101230000.png)

对于训练集中的所有样本都满足𝑓 𝑐(𝒙; 𝒘 𝑐∗) > 𝑓 𝑐̃ (𝒙, 𝒘 𝑐∗̃ ), ∀𝑐 ̃ ≠ 𝑐，就称训练集是线性可分的。

## Logistic 回归

现实中大多的训练集不是线性可分的，所以要引入非线性函数g：

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007101810479.png" alt="image-20221007101810479" style="zoom:67%;" />

其中 𝑔(⋅) 通常称为激活函数，其作用是把线性函数的值域从实数区间“挤压”到了 (0, 1) 之间，可以用来表示概率。

对于Logistic 回归，使用 Logistic 函数来作为激活函数。

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007101951330.png" alt="image-20221007101951330" style="zoom:67%;" />

这里 𝒙 = [𝑥 1, ⋯ , 𝑥 𝐷, 1] T和 𝒘 = [𝑤 1, ⋯ , 𝑤 𝐷, 𝑏] T分别为 𝐷 + 1 维 的**增广特征向量**和**增广权重向量**。

---

Logistic 回归采用交叉熵作为损失函数，并使用梯度下降法来对参数进行优化。

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007102728627.png" alt="image-20221007102728627" style="zoom:67%;" />

对每个样本 𝒙 (𝑛)进 行预测，输出其标签为 1 的后验概率。

损失函数为：

ℛ(𝒘) =<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007102912404.png" alt="image-20221007102912404" style="zoom:50%;" />

![image-20221007103146221](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007103146221.png)

## Softmax 回归

Softmax 回归，也称为多项或多类的 Logistic 回归，是 Logistic 回归在多分类问题上的推广。

给定一个样本 𝒙，Softmax 回归预测的属于类别 𝑐 的条件概率为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007104450993.png" alt="image-20221007104450993" style="zoom: 67%;" />

Softmax 回归的决策函数可以表示为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007104606241.png" alt="image-20221007104606241" style="zoom:67%;" />

---

Softmax 回归使用交叉熵损失函数来学 习最优的参数矩阵 𝑾

用C维的one-hot向量来表示类别标签y

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007105300571.png" alt="image-20221007105300571" style="zoom:67%;" />

Softmax 回归模型的风险函数为

ℛ(𝑾)<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007105451751.png" alt="image-20221007105451751" style="zoom:67%;" />

采用梯度下降法，Softmax 回归的训练过程为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007105539788.png" alt="image-20221007105539788" style="zoom:67%;" />

> 要注意的是，Softmax 回归中使用的 𝐶 个权重向量是冗余的，即对所有的 权重向量都减去一个同样的向量 𝒗，不改变其输出结果．因此，Softmax 回归往往需要使用正则化来约束其参数．此外，我们还可以利用这个特性来避免计算 Softmax 函数时在数值计算上溢出问题．

## 感知器

感知器是最简单的神经网络，只有一个神经元。

对于二分类问题，，感知器学习算法试图找到一组参数 𝒘 ∗ ，使得对于每个样本 (𝒙 (𝑛) , 𝑦 (𝑛) ) 有

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007110030304.png" alt="image-20221007110030304" style="zoom:67%;" />

感知器的学习算法是一种错误驱动的在线学习算法，先初始化一个权重向量w，每次分错一个样本 (𝒙, 𝑦)时，即 𝑦𝒘 T 𝒙 < 0，就用这个样本来更新权重．

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007110206884.png" alt="image-20221007110206884" style="zoom:67%;" />

因此感知器的损失函数为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007110256188.png" alt="image-20221007110256188" style="zoom:67%;" />

采用随机梯度下降，其每次更新的梯度为

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007110328910.png" alt="image-20221007110328910" style="zoom:67%;" />

算法流程：

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007110513236.png" alt="image-20221007110513236" style="zoom:67%;" />

虽然感知器在线性可分的数据上可以保证收敛，但其存在以下不足：

1. 在数据集线性可分时，感知器虽然可以找到一个超平面把两类数据分开，但并不能保证其泛化能力．
2. 感知器对样本顺序比较敏感．每次迭代的顺序不一致时，找到的分割超平面也往往不一致．
3. 如果训练集不是线性可分的，就永远不会收敛．

### 参数平均感知器

为了提高感知器的鲁棒性和泛化能力，将感知器学习过程中更新的所有权重向量𝒘𝑘保存下来，并赋予每个wk一个置信系数 𝑐𝑘，ck等于wk到更新w(k+1)之间间隔的迭代次数，最终的分类结果通过这 𝐾 个不同权重的感知器投票决定，这个模型也称为投票感知器。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007112743763.png" alt="image-20221007112743763" style="zoom:67%;" />

投票感知器虽然提高了感知器的泛化能力，但是需要保存 𝐾 个权重向量． 在实际操作中会带来额外的开销．因此，人们经常会使用一个简化的版本，通 过使用“参数平均”的策略来减少投票感知器的参数数量，也叫作平均感知器。

<img src="/Users/eumenides/Library/Application Support/typora-user-images/image-20221007112808776.png" alt="image-20221007112808776" style="zoom:67%;" />

这个方法非常简单，只 需要在算法3.1中增加一个 𝒘̄，并且在每次迭代时都更新 𝒘̄：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007112859741.png" alt="image-20221007112859741" style="zoom:67%;" />

总的算法流程如下：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007112955901.png" alt="image-20221007112955901" style="zoom:67%;" />



## 支持向量机

支持向量机是一个经典的二分类算法， 其找到的分割超平面具有更好的鲁棒性，因此广泛使用在很多任务上，并表现出了很强优势。

如果两类样本是线性可分的，即存在一个超平面

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007114943860.png" alt="image-20221007114943860" style="zoom:67%;" />

将两类样本分开，每个样本到超平面的距离为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115039509.png" alt="image-20221007115039509" style="zoom:67%;" />

定义间隔（Margin）𝛾 为整个数据集 𝐷 中所有样本到分割超平面的最短距离：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115123630.png" alt="image-20221007115123630" style="zoom:67%;" />

支持向量机的目标是寻找一个超平面 (𝒘 ∗, 𝑏 ∗) 使得 𝛾 最大，即

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115153474.png" alt="image-20221007115153474" style="zoom:67%;" />

由于同时缩放 𝒘 → 𝑘𝒘 和 𝑏 → 𝑘𝑏 不会改变样本 𝒙 (𝑛) 到分割超平面的距离，限制 ‖𝒘‖ ⋅ 𝛾 = 1，则公式等价于：

![image-20221007115359877](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115359877.png)

![image-20221007115412371](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115412371.png)

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007115526437.png" alt="image-20221007115526437" style="zoom: 50%;" />

### 参数学习

见书本71页3.5.1节

### 核函数

支持向量机还有一个重要的优点是可以使用核函数隐式地将样本从原始特征空间映射到更高维的空间，并解决原始特征空间中的线性不可分问题。

比如在一个变换后的特征空间 𝜙 中，支持向量机的决策函数为

![image-20221007121644638](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007121644638.png)

常用的核函数有：[线性](https://baike.baidu.com/item/线性?fromModule=lemma_inlink)核函数，[多项式核函数](https://baike.baidu.com/item/多项式核函数?fromModule=lemma_inlink)，径向基核函数（高斯），Sigmoid核函数和[复合核](https://baike.baidu.com/item/复合核?fromModule=lemma_inlink)函数，[傅立叶级数](https://baike.baidu.com/item/傅立叶级数/7649046?fromModule=lemma_inlink)核，B样条核函数和张量积核函数等

### 软间隔

在支持向量机的优化问题中，约束条件比较严格．如果训练集中的样本在特 征空间中不是线性可分的，就无法找到最优解．为了能够容样本，我们可以引入松弛变量𝜉 ，将优化问题变为

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007122348119.png" alt="image-20221007122348119" style="zoom:67%;" />

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007122403420.png" alt="image-20221007122403420" style="zoom:67%;" />

其中参数 𝐶 > 0 用来控制间隔和松弛变量惩罚的平衡．引入松弛变量的间隔称为软间隔，公式可以表示为经验风险 + 正则化项的形式：

![image-20221007122703108](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007122703108.png)

软间隔支持向量机的参数学习和原始支持向量机类似，其最终决策函数也 只和支持向量有关，即满足 1 − 𝑦 (𝑛) (𝒘 T 𝒙 (𝑛) + 𝑏) − 𝜉 𝑛 = 0 的样本．

# 前馈神经网络

## 神经元

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007123643510.png" alt="image-20221007123643510" style="zoom:67%;" />

一个神经元就是感知器。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007123721839.png" alt="image-20221007123721839" style="zoom:67%;" />

计算出加权和z后，往往需要经过激活函数

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007123730375.png" alt="image-20221007123730375" style="zoom:67%;" />

激活函数在神经元中非常重要的．为了增强网络的表示能力和学习能力，激活函数需要具备以下几点性质：

1. 连续并可导（允许少数点上不可导）的非线性函数，可导的激活函数 可以直接利用数值优化的方法来学习网络参数。
2. 激活函数及其导函数要尽可能的简单，有利于提高网络计算效率．
3. 激活函数的导函数的值域要在一个合适的区间内，不能太大也不能太小，否则会影响训练的效率和稳定性．

常见的激活函数有：

- Sigmoid 型函数

  <img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007123921850.png" alt="image-20221007123921850" style="zoom:67%;" />

- Tanh 函数

  <img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007123945574.png" alt="image-20221007123945574" style="zoom:67%;" />

- Hard-Logistic 函数和 Hard-Tanh 函数

  Logistic 函数和 Tanh 函数都是 Sigmoid 型函数，，但是计算开销较大，可以使用分段函数来近似

  <img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007124101271.png" alt="image-20221007124101271" style="zoom:67%;" />

- ReLU 函数（多用）

  <img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007124134691.png" alt="image-20221007124134691" style="zoom:67%;" />

  计算高效，一定程度缓解了神经网络的梯度消失问题，加速梯 度下降的收敛速度。但ReLU 函数的输出是非零中心化的，如果参数在一次不恰当的更新后，第一个隐藏层中的某个 ReLU 神经元在 所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是 0。ReLU 神经元在训练时比较容易“死亡”。

  为了避免这个问题，带泄露的 ReLU、带参数的 ReLU、ELU 函数、Softplus 函数等变式relu也常常被考虑。

  <img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007124536065.png" alt="image-20221007124536065" style="zoom:67%;" />

## 前馈神经网络

给定一组神经元，我们可以将神经元作为节点来构建一个网络，前馈神经网络也经常称为多层感知器。各神经元分别属于不同的 前一层神经元的信号，并产生信号输出到下一层．第 0 层称为输入层，最后一层称 为输出层，其他中间层称为隐藏层．整个网络中无反馈，信号从输入层向输出层 单向传播，可用一个有向无环图表示．

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007124954227.png" alt="image-20221007124954227" style="zoom:67%;" />

前馈神经网络通过不断迭代下面公式进行信息传播：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007125036801.png" alt="image-20221007125036801" style="zoom:67%;" />

前馈神经网络具有很强的拟合能力，常见的连续非线性函数都可以用前馈神经网络来近似。

梯度下降法需要计算损失函数对参数的偏导数，如果通过链式法则逐一对 每个参数进行求偏导比较低效．在神经网络的训练中经常使用反向传播算法来 高效地计算梯度。

## 反向传播算法

见书本92页4.4节

算法流程如下：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007125541098.png" alt="image-20221007125541098" style="zoom:67%;" />

## 自动梯度计算

自动计算梯度的方法可以分为以下三类：数值微分、符号微分和自动微分。

### 数值微分

数值微分（Numerical Differentiation）是用数值方法来计算函数 𝑓(𝑥) 的导 数．函数 𝑓(𝑥) 的点 𝑥 的导数定义为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131110138.png" alt="image-20221007131110138" style="zoom:67%;" />

数值微分方法非常容易实现，但找到 一个合适的扰动 Δ𝑥 却十分困难．如果 Δ𝑥 过小，会引起数值计算问题，比如舍入 误差；如果 Δ𝑥 过大，会增加截断误差，使得导数计算不准确。

在实际应用，经常使用下面公式来计算梯度，可以减少截断误差。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131201123.png" alt="image-20221007131201123" style="zoom:67%;" />

### 符号微分

符号计算也叫代数计算，是指用计算机来处理带有变量的数学表达式．出都是数学表达式，一般包括对数学表达式的化简、因式分解、微分、积分、解代 数方程、求解常微分方程等运算．

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131321334.png" alt="image-20221007131321334" style="zoom:67%;" />

### 自动微分

自动微分的基本原理是所有的数值计算可以分解为一些基本操作， 包含 +, −, ×, / 和一些初等函数 exp, log, sin, cos 等，然后利用链式法则来自动计算一 个复合函数的梯度。

这里举一个常见的复合函数的例子来说明自动微分的过程。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131500656.png" alt="image-20221007131500656" style="zoom:67%;" />

首先，我们将复合函数 𝑓(𝑥; 𝑤, 𝑏) 分解为一系列的基本操作，并构成一个计算图。

计算图中的每个非叶子节点表示一个基本操作，每个叶子节点为一个输入变量或常量

![image-20221007131634968](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131634968.png)

![image-20221007131719437](/Users/eumenides/Library/Application Support/typora-user-images/image-20221007131719437.png)

---

按照计算导数的顺序，自动微分可以分为两种模式：前向模式和反向模式。反向模式和反向传播的计算梯度的方式相同。

![image-20221007131909116](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007131909116.png)



## 优化问题

神经网络的参数学习比线性模型要更加困难，主要原因有两点：

1. 非凸优化问题。损失函数是非凸的，求解会进入局部最优解。
2. 梯度消失问题。误差从输出层反向传播时，在每一层都要乘以该层的激活函数的导数，对于Logistic或者Tanh激活函数，饱和区的导数接近于 0，当网络层数很深时，梯度就会不停衰减，甚至消失，使得整个网络很难训练．造成梯度消失。



# 激活函数

**ReLU激活函数容易造成神经元死亡。**如果参数在一次不恰当的更新后，第一个隐藏层中的某个 ReLU 神经元在 所有的训练数据上都不能被激活，那么这个神经元自身参数的梯度永远都会是 0，在以后的训练过程中永远不能被激活．这种现象称为死亡 ReLU 问题（Dying ReLU Problem），并且也有可能会发生在其他隐藏层．

---

**Swish是一种自门控（Self-Gated）激活函数**，定义为swish(𝑥) = 𝑥𝜎(𝛽𝑥)，其中 𝜎(⋅) 为 Logistic 函数，𝛽 为可学习的参数或一个固定超参数．𝜎(⋅) ∈ (0, 1) 可 以看作一种软性的门控机制．当 𝜎(𝛽𝑥) 接近于 1 时，门处于“开”状态，激活函数的 输出近似于 𝑥 本身；当 𝜎(𝛽𝑥) 接近于 0 时，门的状态为“关”，激活函数的输出近似 于 0．

---

**GELU**（Gaussian Error Linear Unit，高斯误差线性单元）也是一种通过门控机制来调整其输出值的激活函数，和 Swish 函数比较类似。

# 残差网络

网络层数增多一般会伴着下面几个问题

1. 计算资源的消耗（GPU集群）
2. 模型容易过拟合（海量数据，配合Dropout、正则化等方法）
3. 梯度消失/梯度爆炸问题的产生（Batch Normalization）

但在真实的训练过程中，随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，继续增加网络深度，loss反而会增大。发生了**退化**（degradation）现象。当网络退化时，浅层网络能够达到比深层网络更好的训练效果，如果把低层的特征传到高层，那么效果应该至少不比浅层的网络效果差，比如可以在VGG-100的98层和14层之间添加一条直接映射（Identity Mapping）来达到此效果。基于这种使用直接映射来连接网络不同层直接的思想，残差网络应运而生。

残差网络由一系列残差块组成，一个残差块可以用表示为：

![image-20221006104235840](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/06/image-20221006104235840.png)

残差块分成两部分直接映射部分和残差部分，x_l是直接映射，F(x_l,W_l)是残差部分，一般两个或者三个卷积操作构成。如下图所示

![img](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/06/v2-bd76d0f10f84d74f90505eababd3d4a1_1440w.png)



# 注意力机制

谷歌团队近期提出的用于生成词向量的BERT算法在NLP的11项任务中取得了效果的大幅提升，BERT算法最重要的部分便是**Transformer**。

Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。作者考虑Attention机制的原因是考虑到RNN（或者LSTM，GRU等）的计算限制为是顺序的，这种机制带来了两个问题：

1. 时间片 t 的计算依赖 t−1 时刻的计算结果，这样限制了模型的并行能力；
2. 顺序计算的过程中信息会丢失，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。

Attention的计算过程可以分为7步：

1. 将单词转化成嵌入向量
2. 根据嵌入向量得到q、k、v三个向量
3. 为每个向量计算一个score： score=q⋅k 
4. 为了梯度的稳定，Transformer使用了score归一化，即除以根号dk
5. 对score施以softmax激活函数
6. softmax点乘Value值 v ，得到加权的每个输入向量的评分 
7. 相加之后得到最终的输出结果 z ： z=∑v 。

注意力表达式为：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/image-20221007220427544.png" alt="image-20221007220427544" style="zoom:67%;" />

**Transformer模型架构**

Transformer模型总体的样子如下图所示：总体来说，还是和Encoder-Decoder模型有些相似，左边是Encoder部分，右边是Decoder部分。

![crYvRd](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/07/crYvRd.jpg)

**Encoder**：输入是单词的Embedding，再加上位置编码，然后进入一个统一的结构，这个结构可以循环很多次（N次），也就是说有很多层（N层）。每一层又可以分成Attention层和全连接层，再额外加了一些处理，比如Skip Connection，做跳跃连接，然后还加了Normalization层。

**Decoder**：第一次输入是前缀信息，之后的就是上一次产出的Embedding，加入位置编码，然后进入一个可以重复很多次的模块。该模块可以分成三块来看，第一块也是Attention层，第二块是cross Attention（Encoder-Decoder Attention），不是Self-Attention，第三块是全连接层。也用了跳跃连接和Normalization。

**输出**：最后的输出要通过Linear层（全连接层），再通过softmax做预测。



具体细节见https://luweikxy.gitbook.io/machine-learning-notes/self-attention-and-transformer#decoder-mo-kuai



# AlexNet

- 使用relu可以加快收敛速度，减少计算量，传统的sigmoid激活函数导入易进入0，出现梯度消失现象

- LRN归一化又称局部响应归一化，限制Relu激活函数的值域

  ![image-20221012212520923](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/12/image-20221012212520923.png)

- Overlapping Pooling 覆盖池化

  pool_size和stride一般是相等的，但当stride小于pool_size时就会出现覆盖，理论上可以保留更多的信息

- Data Augmentation

  通过平移，旋转，拉长，变形，来防止过拟合

- Dropout正则化（增加扰动因子，防止过拟合）

![image-20221012213442317](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/12/image-20221012213442317.png)

  



# 降维

## t-SNE

![Snipaste_2022-10-13_19-04-13](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/13/Snipaste_2022-10-13_19-04-13.jpg)

关于高维分布pij中*σ* 该怎么确定

![image-20221013191619476](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/13/image-20221013191619476.png)

## UMAP

![image-20221013193027989](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/13/image-20221013193027989.png)

UMAP相比t-SNE的`优势`有：

- 保留更多的局部结构，降维效果更好
- 计算时间短，高效（主要）
- 可以直接使用在更高的维度上（对于高维数据，需要先PCA降维再使用t-SNE）

UMAP与t-SNE的`区别`有：

- 去掉了分母正则项，降低了复杂度

- 引入了距离点i最近的点pi，如果一个点i距离所有的点都很远，那么i点与所有点做邻居的条件概率都接近0，会造成图的不连接。引入pi保证至少有一个点j与点i做邻居的条件概率为1，保证了图的连通性

- 增加了超参数a、b，使得低维分布函数可以近似为分段函数，使高维距离近的点，在低维也能近。其中min_dist为超参数，设置后可以通过曲线拟合a、b值

  ![image-20221013195721629](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/13/image-20221013195721629.png)

- t-SNE的损失函数当高维距离很大，低维距离很小时，出现了梯度消失，造成了信息的丢失，因此t-SNE不能保证高维距离远的点在低维也很远。UMAP则更加全面

  ![image-20221013200107065](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/13/image-20221013200107065.png)

# 损失函数

## triplet loss

**triplet**：一个三元组，这个三元组是这样构成的：从训练数据集中随机选一个样本，该样本称为**Anchor**，然后再随机选取一个和Anchor (记为x_a)属于同一类的样本和不同类的样本,这两个样本对应的称为**Positive** (记为x_p)和**Negative** (记为x_n)，由此构成一个三元组。

**triplet loss：**针对三元组中的每个元素（样本），训练一个参数共享或者不共享的网络，得到三个元素的特征表达，分别记为：f(x_a)、f(x_p)、f(x_n)。通过学习，让x_a和x_p特征表达之间的距离尽可能小，而x_a和x_n的特征表达之间的距离尽可能大，并且要让x_a与x_n之间的距离和x_a与x_p之间的距离之间有一个最小的间隔alpha。

**目标函数**：距离用欧式距离度量，+表示[]内的值大于零的时候，取该值为损失，小于零的时候，损失为零。 

![39aQqN](https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/17/39aQqN.jpg)

## Cross Entropy

分类问题常用的损失函数为交叉熵( Cross Entropy Loss)。

**信息量**：信息量的大小和事件发生的概率成反比。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/17/aGpxrD.jpg" alt="aGpxrD" style="zoom:200%;" />

**信息熵**： 信息量度量的是一个具体事件发生所带来的信息，而信息熵是考虑该随机变量的所有可能取值，即所有可能发生事件所带来的信息量的期望。

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/17/HpDDCa.jpg" alt="HpDDCa" style="zoom:200%;" />

**相对熵（KL散度）**：p(x)为样本真实分布，q(x)为预测分布，**KL散度越小，表示p(x) 与q(x)的分布更加接近，可以通过反复训练q(x)来使q(x) 的分布逼近p(x)。**

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/17/3R3YQ8.jpg" alt="3R3YQ8" style="zoom:200%;" />

**交叉熵**：

<img src="https://raw.githubusercontent.com/0Eumenides/upic/main/2022/10/17/NJKDzG.jpg" alt="NJKDzG" style="zoom:200%;" />

