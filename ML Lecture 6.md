# 神经网络概览 Artifcial Neural Network
## 人工神经网络结构

一个神经网络包含以下四个组成：
- 输入层（Input layer）
- 一个或多个隐藏层（Hidden layers）
- 输出层（Output layer）
- 各层之间的连接（Connections）

![](Assets/Pasted%20image%2020251211231545.png)

在网络中，**信号在神经元之间被处理并传递**：上一层神经元的输出作为下一层神经元的输入，通过连接权重加权后再送入激活函数。

### 激活函数 Activation functions 

激活函数用于在神经元中**引入非线性**，常见的有：

- Sigmoid 函数，记作 `sigm`  
  $$\mathrm{sigm}(x) = \frac{1}{1 + e^{-x}}$$

- 双曲正切函数（tanh），记作 `tanh`  
  $$\tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

- 修正线性单元（Rectified Linear Unit, ReLU），记作 `ReLU`  
  $$
  \mathrm{ReLU}(x) =
  \begin{cases}
  x, & x > 0 \\
  0, & \text{otherwise}
  \end{cases}
  $$

### 神经元内部的计算与激活 Processes and activation within a neuron

一个神经元通常包含：

- 权重向量 $w$
- 偏置项 $b$
- 激活函数（如 sigmoid、tanh、ReLU 等）

其基本计算步骤包括：

1. **乘法（Multiplication）**：输入与权重相乘  
2. **加法（Addition）**：乘积求和并加上偏置  
3. **激活（Activation）**：对线性结果应用激活函数

假设输入为 $i_1 = 0.8$。

1. 第一层神经元：

   - 权重：$w_1 = 0.25$  
   - 偏置：$b_1 = 0.12$  
   - 线性部分：
     $$
     w_1 i_1 + b_1 = 0.25 \times 0.8 + 0.12 = 0.32
     $$
   - 激活输出（采用 sigmoid）：
     $$
     \mathrm{sigm}(0.32) = \frac{1}{1 + e^{-0.32}} \approx 0.579
     $$

   记该输出为新的输入 $i_2 = 0.579$。

2. 下一层神经元：

   - 权重：$w_2 = 0.14$  
   - 偏置：$b_2 = -0.3$  
   - 线性部分：
     $$
     w_2 i_2 + b_2 = 0.14 \times 0.579 - 0.3 \approx -0.2189
     $$
   - 激活输出（同样使用 sigmoid）：
     $$
     \mathrm{sigm}(-0.2189) = \frac{1}{1 + e^{0.2189}} \approx 0.4455
     $$

通过这样的“乘法 → 加法 → 激活”过程，信号在网络中一层层传播并被非线性变换。

![](Assets/Pasted%20image%2020251211231708.png)


> 在神经网络中引入激活函数（Activation Function）来增加非线性，其核心意义可以总结为一句话：**赋予神经网络处理复杂问题的能力（即提高模型的表达能力）。**

> 如果没有激活函数，无论神经网络有多少层，它本质上都只是一个线性回归模型。


> 如果不用激活函数：
    神经网络的每一层都在做线性变换（$y = wx + b$）。线性函数的线性组合仍然是线性函数。
    假设你有一个两层的网络：
    1. 第一层输出：$z_1 = w_1 x + b_1$
    2. 第二层输出：$y = w_2 z_1 + b_2$
    代入后你会发现：
    
    $$y = w_2(w_1 x + b_1) + b_2 = (w_2 w_1)x + (w_2 b_1 + b_2) = W'x + B'$$
    
 >   这说明，**没有非线性激活函数，多层网络在数学上等价于单层网络**。你费劲搭建的“深度”网络会塌缩成一个简单的线性模型，无法学习复杂的特征。
>- 引入激活函数后：
    每一层的输出经过非线性变换（如 ReLU, Sigmoid），打破了线性的叠加。使得神经网络能够通过层层叠加，逼近任意复杂的函数。


---
## 向前传播 Forward Propagation

向前传播就是把输入数据从输入层依次传到输出层、并在每一层完成“加权求和 + 加偏置 + 过激活函数”的计算过程：对某一层的每个神经元，先把上一层各神经元的输出乘以对应权重后相加，再加上一个偏置项得到 $z$，然后把 $z$ 输入到激活函数（如 sigmoid、ReLU 等）得到该神经元的输出，这些输出再作为下一层的输入，如此一层一层往前算，直到输出层得到最终预测结果，这整个从输入到输出的计算流程就叫前向传播。

![](Assets/Pasted%20image%2020251211233905.png)

- 输入向量 Input: `[0.35, 0.9]`
- 期望输出 Expected output: `[0.5]`
- 激活函数 Activation function: **Sigmoid 函数 Sigmoid function**

> 注意：**激活函数 Activation function** 不作用于 **输入层 Input layer**。

1. 第一步：计算隐藏层神经元 $h_1$


-  加权求和（Weighted Sum, $ih_1$）：
    我们将输入值乘以对应的权重，然后相加。
    - 输入 $i_1 (0.35)$ 乘以权重 $w_1 (0.1)$。
    - 输入 $i_2 (0.9)$ 乘以权重 $w_3 (0.8)$。
    - 计算：$ih_1 = 0.35 \times 0.1 + 0.9 \times 0.8 = 0.035 + 0.72 = 0.755$。
        
- 激活处理（Activation, $h_1$）：
    将求和结果 $0.755$ 代入 Sigmoid 函数。
    - 计算：$h_1 = f(0.755) = \frac{1}{1 + e^{-0.755}} \approx 0.68$。
    - **结果**：隐藏层第一个神经元 $h_1$ 的输出值为 **0.68**。


2. 第二步：计算隐藏层神经元 $h_2$

- 加权求和（Weighted Sum, $ih_2$）：
    同样地，计算连接到 $h_2$ 的输入加权和。
    - 输入 $i_1 (0.35)$ 乘以权重 $w_2 (0.4)$。
    - 输入 $i_2 (0.9)$ 乘以权重 $w_4 (0.6)$。
    - 计算：$ih_2 = 0.35 \times 0.4 + 0.9 \times 0.6 = 0.14 + 0.54 = 0.68$。

- 激活处理（Activation, $h_2$）：
    将求和结果 $0.68$ 代入 Sigmoid 函数。
    - 计算：$h_2 = f(0.68) = \frac{1}{1 + e^{-0.68}} \approx 0.6637$。
    - **结果**：隐藏层第二个神经元 $h_2$ 的输出值为 **0.6637**。
        
3. 第三步：计算输出层神经元 $o_1$

- **加权求和（Weighted Sum, $io_1$）**：
    - 隐藏层输出 $h_1 (0.68)$ 乘以权重 $w_5 (0.3)$。
    - 隐藏层输出 $h_2 (0.6637)$ 乘以权重 $w_6 (0.9)$。
    - 计算：$io_1 = 0.68 \times 0.3 + 0.6637 \times 0.9 = 0.204 + 0.59733 = 0.80133 \approx 0.801$。

- 激活处理（Activation, $o_1$）：
    将求和结果 $0.801$ 代入 Sigmoid 函数。
    * 计算：$o_1 = f(0.801) = \frac{1}{1 + e^{-0.801}} \approx 0.69$。

![](Assets/Pasted%20image%2020251211234643.png)

---
## 向后传播 Backward Propagation

向后传播（Backpropagation）是神经网络训练的核心算法，本质上是一种基于微积分链式法则的高效梯度计算机制。在神经网络完成前向传播并计算出预测值与真实标签之间的误差（Loss）后，向后传播会将这个误差从输出层反向传递回输入层。在这个过程中，算法逐层计算损失函数相对于网络中每一个权重和偏置的偏导数（即梯度），从而精确量化每个参数对最终总误差的“贡献程度”。形象地说，它就像是一个自动化的“归责系统”，通过反向推导找出导致预测偏差的具体原因，明确告诉网络中的每一个参数：为了让结果更准确，你应该变大一点还是变小一点，以及变化的幅度需要多大。

**向后传播的目的和作用**在于指导神经网络进行参数优化，从而实现模型的“学习”。它为梯度下降等优化算法提供了至关重要的导航信息——梯度。如果没有向后传播，优化器就无法知道参数更新的正确方向，网络也就无法从错误中吸取教训。通过向后传播计算出的梯度，优化算法能够有的放矢地微调成千上万个权重参数，使得模型在下一次前向传播时的预测误差减小。随着“前向预测—反向求导—参数更新”这一循环的反复进行，神经网络的性能逐渐收敛，最终具备处理复杂任务的预测能力。

### 损失函数 Loss function

**单个神经元示例 Single neuron example**

- 权重 Weight: $w_1 = 0.25$
- 偏置 Bias: $b_1 = 0.12$
- 输入到神经元的值 Input to the neuron: $i_1 = 0.8$
- 神经元输出 Output: $o_1 = 0.579$

若 **真实值 Ground truth**（即期望输出 expected value）设为 $t = 1$，  
则要回答：$o_1 = 0.579$ 是否是对目标 $t = 1$ 的一个“好”的估计？

这就需要用 **损失函数 Loss function** 来量化预测效果。


设共有 $N$ 个样本，输出为 $o_i$，真实值为 $t_i$。

1. **平均绝对误差 Mean absolute error**（MAE）  
   $$
   E_{\text{MAE}} = \frac{1}{N} \sum_{i=1}^{N} \lvert o_i - t_i \rvert
   $$
   本例只有一个样本：
   $$
   E_{\text{MAE}} = \lvert 0.579 - 1 \rvert = 0.421
   $$

2. **平均平方误差 Mean squared error**（MSE）  
   （课件后面使用的是带 $\frac{1}{2}$ 的形式，便于求导）  
   $$
   E_{\text{MSE}} = \frac{1}{N} \sum_{i=1}^{N} (o_i - t_i)^2
   $$
   本例：
   $$
   E_{\text{MSE}} = (0.579 - 1)^2 \approx 0.1772
   $$

3. **均方根误差 Root mean squared error**（RMSE）  
   $$
   E_{\text{RMSE}} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (o_i - t_i)^2}
   $$
   本例：
   $$
   E_{\text{RMSE}} = \sqrt{(0.579 - 1)^2} = 0.421
   $$


### 梯度下降算法 Gradient descent algorithm

- （最快）下降方向是 **梯度 Gradient** 的相反方向。  
- 按小步迭代，逐步逼近最小值。  
- 步长由 **学习率 Learning rate** 控制。

最简单情形下的更新公式：

$$
\Delta w = - \eta \frac{\partial E(w)}{\partial w}
$$

即：
- 先计算损失对权重的偏导数 **梯度 Gradient**；
- 再沿负梯度方向更新权重，使损失减小。

逐项计算：

1. 误差对输出的导数：
   $$
   \frac{\partial E}{\partial o_1}
     = \frac{\partial}{\partial o_1} \frac{(o_1 - \text{target})^2}{2}
     = o_1 - \text{target}
     = 0.69 - 0.5 = 0.19
   $$

2. 输出对加权和 $io_1$ 的导数（Sigmoid 的导数）：
   $$
   \frac{\partial o_1}{\partial io_1}
     = \frac{\partial}{\partial io_1} \left( \frac{1}{1 + e^{-io_1}} \right)
     = o_1 (1 - o_1)
     = 0.69 (1 - 0.69) \approx 0.2139
   $$

3. 加权和对权重 $w_5$ 的导数：
   $$
   io_1 = h_1 w_5 + h_2 w_6
   $$
   $$
   \frac{\partial io_1}{\partial w_5} = h_1 = 0.68
   $$

于是：
$$
\frac{\partial E}{\partial w_5}
  = 0.19 \times 0.2139 \times 0.68 \approx 0.0276
$$

权重更新（梯度下降）：
$$
\Delta w_5 = \eta \frac{\partial E}{\partial w_5}
           = 1 \times 0.0276 = 0.0276
$$
$$
w_5^{\text{new}} = w_5^{\text{old}} - \Delta w_5
                 = 0.3 - 0.0276
                 \approx 0.2724
$$

![](Assets/Pasted%20image%2020251212110403.png)



对 $w_6$ 使用同样的链式法则：
$$
\frac{\partial E}{\partial w_6}
  = \frac{\partial E}{\partial o_1}
    \cdot \frac{\partial o_1}{\partial io_1}
    \cdot \frac{\partial io_1}{\partial w_6}
$$

这里：
$$
\frac{\partial io_1}{\partial w_6} = h_2 = 0.6637
$$

最终可以得到新的 $w_6$：
$$
w_6^{\text{new}} \approx 0.8731
$$

### 反向传播：更新隐藏层权重 Updating hidden-layer weights

现在考虑隐藏层上的权重，例如 $w_1$（从 $i_1$ 到 $h_1$）。

同样问题：$w_1$ 的变化如何影响总误差 $E$？

根据 **链式法则 Chain rule**：

$$
\frac{\partial E}{\partial w_1}
  = \frac{\partial E}{\partial o_1}
    \cdot \frac{\partial o_1}{\partial io_1}
    \cdot \frac{\partial io_1}{\partial h_1}
    \cdot \frac{\partial h_1}{\partial ih_1}
    \cdot \frac{\partial ih_1}{\partial w_1}
$$

通常把
$$
\frac{\partial E}{\partial o_1}
  \cdot \frac{\partial o_1}{\partial io_1}
  = \frac{\partial E}{\partial io_1}
  = \delta_o
$$
称为输出层的 **误差项 Delta**，记作 $\delta_o$。

各项分别为：

- $\frac{\partial io_1}{\partial h_1} = w_5$
- $h_1 = \dfrac{1}{1 + e^{-ih_1}}$，所以
  $$
  \frac{\partial h_1}{\partial ih_1}
    = h_1 (1 - h_1)
  $$
- $ih_1 = i_1 w_1 + i_2 w_3$，因此
  $$
  \frac{\partial ih_1}{\partial w_1} = i_1
  $$

因此可以简化为：
$$
\frac{\partial E}{\partial w_1}
  = \delta_o \, w_5 \, h_1 (1 - h_1) \, i_1
$$

代入数值（课件给出的数值）：
$$
\Delta w_1
  = \eta \frac{\partial E}{\partial w_1}
  = 1 \times 0.19 \times 0.2139 \times 0.2724
    \times 0.68 \times (1 - 0.68) \times 0.35
  \approx 8.43 \times 10^{-4}
$$
$$
w_1^{\text{new}}
  = w_1^{\text{old}} - \Delta w_1
  = 0.1 - 8.43 \times 10^{-4}
  \approx 0.0991
$$

同理，可以得到 $w_2, w_3, w_4$ 的更新值。


### 一次前向 + 反向传播之后 Forward and backward propagation

一次完整的前向 + 反向更新后，各权重大致变为：

- $w_1 \approx 0.0991$
- $w_2 \approx 0.3971$
- $w_3 \approx 0.7976$
- $w_4 \approx 0.5926$
- $w_5 \approx 0.2724$
- $w_6 \approx 0.8731$

用这些新权重重新进行一次 **前向传播 Forward propagation**，得到新的输出：

$$
o_1 \approx 0.682
$$

新的误差为：
$$
E = \frac{1}{2} (o_1 - \text{target})^2
  = \frac{1}{2} (0.682 - 0.5)^2
  \approx 0.01656
$$

相比之前的 $E \approx 0.01805$ 更小，说明通过一次 **梯度下降 Gradient descent**，网络在这个样本上的拟合更好了。  
实际训练中需要进行多次迭代 **More iterations** 才能得到更好的结果。

![](Assets/Pasted%20image%2020251212110505.png)


### 预测阶段 Prediction phase

当通过多次训练后，假设权重已经收敛到某个较优解，例如：

- $w_1 = -0.1$
- $w_2 = -0.2$
- $w_3 = 0.6$
- $w_4 = 0.8$
- $w_5 = -0.1$
- $w_6 = 0.2$

此时对同样的输入 `[0.35, 0.9]` 做 **前向传播 Forward propagation**，可以得到：

- 隐藏层输出大致为：
  - $h_1 \approx 0.6236$
  - $h_2 \approx 0.3122$
- 输出层结果：
  $$
  o_1 \approx 0.5002
  $$

因此，网络输出 $o_1$ 已经非常接近目标 $0.5$，可以认为网络在该样本上收敛到一个不错的解。  
这就是训练好的网络在 **预测阶段 Prediction phase** 的工作方式：  
给定输入，通过前向传播直接给出输出，不再进行反向传播和权重更新。

---
## 神经网络训练

### 训练轮次与批量训练 Training epochs and batch training

- 使用一个训练样本进行一次参数更新：  
  - 包含 $1$ 次 **前向传播 Forward propagation**  
  - 和 $1$ 次 **反向传播 Backward propagation**

- 一个 **轮次 Epoch** 的含义：  
  - 对所有训练样本各做一次  
  - 即：**样本数 number of training samples** $\times$ （前向 + 反向传播）

- **批量训练 Batch training**：  
  - 基于**平均损失 averaged loss** 来更新模型参数  
  - 有时会使用 **小批量 Mini-batch** 的方式


### 小批量梯度下降 Mini-batch Gradient Descent（MBGD）

- 在 **批量梯度下降 Batch Gradient Descent, BGD** 中，每次用**所有样本 all samples** 来更新参数。
- 在 **随机梯度下降 Stochastic Gradient Descent, SGD** 中，每次只用**一个样本 a single sample**。
- 在 **小批量梯度下降 Mini-batch Gradient Descent, MBGD** 中，每次使用**一个小批量 mini-batch**。

- 一个 **小批量 mini-batch** 是包含 $M$ 个训练样本的子集，满足 $M < N$（$N$ 为总样本数）。
- $M$ 被称为 **批大小 Batch size**。

#### MBGD 的基本流程

1. 将数据集划分为多个 **小批量 mini-batches**。  
2. 从训练集中取出一个 **小批量 mini-batch**。  
3. 计算该小批量上损失函数的**平均梯度 Mean gradient**。  
4. 根据这个平均梯度更新模型参数。  
5. 对所有小批量重复步骤 2)–4)。

#### MBGD 的超参数与性质 Hyperparameter and properties

在 MBGD 中，$M$ 是影响学习过程的一个 **超参数 Hyperparameter**：

- **较小的 $M$ Small $M$**：行为更接近 **SGD**。  
- **较大的 $M$ Large $M$**：行为更接近 **BGD**。

通常一个比较常见的默认值是 $M = 32$。

MBGD 结合了 BGD 和 SGD 的优点，因此：

- 在一定程度上可以避免收敛到**局部极小值 Local minima**。  
- 需要的**内存 Memory** 较少。  
- **收敛速度 Convergence speed** 较快。


### Adam 优化器 The Adam optimizer

#### 普通梯度下降 Gradient descent

![](Assets/Pasted%20image%2020251212111827.png)

- 使用**固定学习率 Fixed learning rate**，对所有参数都相同。  
- 学习率过小，会导致收敛太慢；  
- 学习率过大，可能震荡甚至发散。  

#### Adam 优化器 Adam optimizer

- Adam 是一种具有**自适应学习率 Adaptive learning rate** 的优化算法。  
- 对不同参数使用不同的、随时间变化的学习率。  
- 它利用梯度的一阶矩和二阶矩的估计：
  - **一阶矩 First moment**：类似于梯度的**均值 Mean**。  
  - **二阶矩 Second moment**：类似于梯度平方的**均值 Mean of squared gradients**。

主要优点：在实践中通常**效率高 Efficient**、收敛速度快，调参相对简单。

### 总结

要得到一个可用的 **神经网络模型 Neural network model**，通常需要以下几个步骤：

1. **给定初始权重和偏置 Initialize weights and bias**  
   - 一种常用方法是**随机初始化 Random initialization**。

2. **根据输入计算输出 Compute output from inputs**  
   - 这一步就是 **前向传播 Forward propagation**。  
   - 输入数据来自**训练集 Training data**。

3. **根据输出与标签的差异定义损失函数 Define loss function**  
   - 计算网络输出与真实标签之间的差异。  
   - 常见损失函数包括：  
     - $L1$ 损失 **L1 loss**  
     - 均方误差损失 **MSE loss**  
     - 交叉熵损失 **Cross-entropy loss** 等。

4. **计算损失函数的梯度 Compute gradients of the loss function**  
   - 然后**利用梯度最小化损失 Minimize gradients and loss**，这一步就是 **反向传播 Back propagation**。  
   - 可选的优化方法很多，例如：  
     - **随机梯度下降 Stochastic Gradient Descent, SGD**  
     - **Adadelta**  
     - **自适应矩估计 Adaptive Moment Estimation, Adam** 等。

5. **当损失足够小 When the loss is small enough**  
   - 保存当前的 **权重 Weights** 和 **偏置 Bias**，作为最终的网络模型。

---
## 神经网络训练的高级主题 Advanced topics in neural network training

### 挑战 Challenges

- **梯度消失 Vanishing gradients**  
  - 在远离输出层的深层网络中，梯度非常小。  
  - 结果：**学习速度变慢 Learning slows down**。

- **梯度爆炸 Exploding gradients**  
  - 网络中的梯度非常大。  
  - 结果：**训练不稳定 Instability in learning**。

### 解决

- **权重初始化 Weight initialization**
- **梯度裁剪 Gradient clipping**
- **激活函数 Activation function**
- **批归一化 Batch normalization**
- **跳跃连接 Skip connection**
- **学习率 Learning rate**


### 欠拟合与过拟合 Underfitting and overfitting

#### 三种拟合情况 Three fitting regimes

- **过拟合 Overfitting**  
  - 在图中，模型在训练集上的决策边界/曲线非常弯曲，过分关注训练数据中的细微模式。  
  - 定义：模型过于关注训练集中的细节和噪声，**Overfitting: the model pays attention to subtle patterns in the training set**。

- **恰当拟合 Good fitting / Right fitting**  
  - 模型既能很好地拟合训练数据，又能对未见数据有良好的泛化能力。

- **欠拟合 Underfitting**  
  - 在图中，模型曲线过于简单，无法捕捉真实关系。  
  - 定义：模型未能学习到训练集中潜在的真实关系，  
    **Underfitting: the model fails to capture the underlying relationships in the training set**。

![](Assets/Pasted%20image%2020251212121029.png)

#### 训练集与测试集上的损失 Loss on training and test sets

| 情况 Case           | 训练集 Training set | 测试集 Test set  |
| ----------------- | ---------------- | ------------- |
| 过拟合 Overfitting   | 低损失 Low loss     | 高损失 High loss |
| 恰当拟合 Good fitting | 低损失 Low loss     | 低损失 Low loss  |
| 欠拟合 Underfitting  | 高损失 High loss    | 高损失 High loss |


### 欠拟合：原因与解决方案 Underfitting, reasons and solutions

**欠拟合的原因及对应解决方案 Causes of underfitting and solutions：**

- **训练数据多样性不足 Limited training data diversity**  
  - 解决：数据采集 **Data acquisition**，收集更多能代表整体分布的数据。

- **特征稀缺或不相关 Feature scarcity and irrelevance**  
  - 解决：**数据预处理 Data preprocessing** 与 **特征选择 Feature selection**，构造和筛选更有信息量的特征。

- **超参数设置不当 Inappropriate hyperparameter setting**  
  - 解决：微调模型的 **超参数 Hyperparameters**，包括  
    **学习率 Learning rate**、**激活函数 Activation function**、**批大小 Batch size** 等。

- **训练轮次不足 Lack of training iterations**  
  - 解决：增加训练时间和训练轮次，直到模型收敛。

- **模型复杂度不足 Insufficient model complexity**  
  - 解决：为神经网络添加更多的 **层 layers** 和/或 **神经元 neurons**。



### 过拟合：原因与解决方案 Overfitting, reasons and solutions

**过拟合的原因及对应解决方案 Causes of overfitting and solutions：**

- **训练样本不足 Insufficient training samples**  
  - 解决：通过数据采集 **Data acquisition** 获取更多样本。

- **数据不平衡 Imbalanced data**  
  - 解决：在不平衡数据上使用加权决策等方法，  
    **Applying weighted decision-making on imbalanced data**。

- **数据泄漏 Data leakage**  
  - 解决：严格保证 **测试集样本 test set samples** 不参与训练过程。

- **超参数设置不当 Inappropriate hyperparameter settings**  
  - 解决：微调模型 **超参数 Hyperparameters**，包括  
    **学习率 Learning rate**、**激活函数 Activation function**、**批大小 Batch size** 等。

- **缺乏正则化 Lack of regularization**  
  - 解决：在训练中加入 **正则化 Regularization**，例如 $L1/L2$ 正则化、Dropout 等。

- **模型过于复杂 Excessive model complexity**  
  - 解决：适当减少网络的 **层数 layers** 和/或 **神经元数量 number of neurons**，简化模型结构。


---
## 缓解过拟合的一些方法 A few methods to improve overfitting

下面介绍几种常见的缓解 **过拟合 Overfitting** 的技术：  
- **正则化 Regularization**  
- **训练集/验证集/测试集划分 Training-validation-test splitting**  
- **早停 Early stopping** 等。

### 正则化 Regularization

在原有损失函数的基础上加入加权的 **正则项 Regularization term** $R(f)$：

$$
\min_f \sum_i E\bigl(f(x_i), y_i\bigr) + \lambda R(f)
$$

- $E(f(x_i), y_i)$：**误差函数 Error function**，刻画预测和真实标签的差异  
- $\lambda$：**正则化系数 Regularization coefficient**，控制数据误差与正则项之间的权衡  
- $R(f)$：与函数 $f$ 的复杂度相关的惩罚项 **复杂度惩罚 penalty on the complexity of $f$**


如果**没有正则项 Without regularization term**，损失函数为：

$$
\min_f \sum_i E\bigl(f(x_i), y_i\bigr)
$$

此时模型可能会变得非常复杂，几乎“记住 memorize” 所有训练样本 $(x_i, y_i)$，导致严重的过拟合。

加入 $R(f)$ 后，可以：

- 惩罚过于复杂的模型  
- 鼓励较平滑、复杂度较低的函数  
- 使模型在**新样本 New data samples** 上具有更好的**泛化能力 Generalization ability**

图示对比了未正则化模型（曲线剧烈波动）和正则化后的模型（更平滑）。

![](Assets/Pasted%20image%2020251212153301.png)


### L1 正则化 L1 Regularization

$R(f)$：仍然表示对模型复杂度的惩罚，使模型能推广到新样本。

**L1 正则化 L1 regularization**，也称为 **Lasso 回归 Lasso regression**：

- 正则项形式：

  $$
  R(f) = \sum_j |\beta_j|
  $$

  其中 $\beta_j$ 为模型参数系数 **model parameter coefficients**。

![](Assets/Pasted%20image%2020251212153346.png)

- 目标是最小化带惩罚的损失函数：

  $$
  \min_f \sum_i E\bigl(f(x_i), y_i\bigr) + \lambda R(f)
  $$

- 图中红色等高线表示原始损失函数的**优化景观 Optimization landscape**，绿色菱形是 L1 约束  
- 交点 $\hat{\beta}$（带帽的 $\beta$）为给定训练集下损失最小的参数：

  - $\hat{\beta}$：**最优参数估计值 Parameter coefficients with minimum loss**

- L1 正则化的一个重要效果：  
  - 会把一些**不重要的参数 Less important coefficients**（例如图中的 $\beta_1$）压缩到 $0$，  
  - 从而实现**特征选择 Feature selection** 和稀疏模型。


## L2 正则化 L2 Regularization

同样，$R(f)$ 仍为复杂度惩罚，使模型能更好推广到新样本。

**L2 正则化 L2 regularization**，也称为 **岭回归 Ridge regression**：

- 正则项形式：

  $$
  R(f) = \sum_j (\beta_j)^2
  $$

- 其中 $\beta_j$ 仍为模型参数系数 **model parameter coefficients**。

![](Assets/Pasted%20image%2020251212153421.png)

- 图中绿色圆形表示 L2 约束区域：$\beta_1^2 + \beta_2^2 \le s$  
- L2 正则化的效果：

  - 将参数向量 $\beta$ **均匀地向原点拉回 Draws $\beta$ back toward origin evenly**  
  - 不会像 L1 那样把参数变为严格的 $0$，但会整体减小参数大小  
  - 通常能得到**较少过拟合 least overfit** 的模型


## 训练集–验证集–测试集划分 Training-validation-test splitting

在训练过程中需要**评估模型 Evaluate the model while training**。  
常见做法是把完整数据集划分为三个集合：

1. **训练集 Training set**  
   - 用来训练模型参数。

2. **验证集 Validation set**  
   - 在训练过程中定期评估模型性能。  
   - 用于选择超参数、判断是否停止训练等。

3. **测试集 Test set**  
   - 在训练和调参结束后，作为**最终评估 Final testing** 使用。  
   - 不参与训练和模型选择。

基本流程：

- 首先从完整数据集 **Full dataset** 随机采样，进行 **训练集–测试集划分 Training-test splitting**。  
- 在训练部分内部，再随机划分出 **验证集 Validation set**。  
- 一般依据验证集的误差曲线来决定是否停止训练：
  - 当验证误差不再下降或开始上升时，停止训练。  
  - 这就是 **早停 Early stopping** 的思想。



### 过拟合与训练曲线 Overfitting – training curve

通过观察 **验证集 Validation set** 的学习曲线，可以进行早停：

- 横轴：训练轮次 **Epochs**  
- 纵轴：损失 **Loss**

随训练进行：

- 训练集损失 Training set loss 通常持续下降。  
- 验证集损失 Validation set loss 先下降、后上升：
  - 前期：模型尚未充分学习，属于 **欠拟合 Underfitting** 区域。  
  - 中间：验证集损失达到最低点，模型处于**最佳拟合 Good fitting**。  
  - 后期：验证集损失开始上升，说明出现 **过拟合 Overfitting**。

**早停 Early stopping**：

- 在验证集损失达到最小值时停止更新模型参数。  
- 即：**The model parameter update stops at the minimum validation loss**。  

常见的训练集 : 验证集 : 测试集划分比例示例：

- $8 : 1 : 1$
- $7 : 1.5 : 1.5$
- $6 : 2 : 2$
- $7 : 2 : 1$

这些比例可以根据数据规模和任务需求灵活调整。

![](Assets/Pasted%20image%2020251212153510.png)
