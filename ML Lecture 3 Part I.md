
---
## ML 模型训练中的基础概念
### 监督学习流程 Supervised Learning Flow
![](Assets/Pasted%20image%2020251205102351.png)

### 超参数与模型参数 Hyperparamters vs. Model Parameters
- 模型参数是模型内部的配置变量，它们的取值由数据来估计得到。
- 模型参数是在训练过程中学习得到的。
	例如：线性回归中的系数 www 和常数项 bbb。


- 超参数是预先设定的参数，它们的取值用来控制学习过程。
- 超参数是由人工在训练开始之前设定的。
    例如：KNN 中的 kkk 等。

### 损失函数 Loss Function
**损失函数**是用来评估模型实际输出与预测输出之间差异的函数。
由损失函数计算得到的数值称为“损失”（loss）。

损失函数有多种形式：
- **Mean squared error**  
	均方误差（MSE）
- **Mean absolute error**  
	平均绝对误差（MAE）
- **Categorical cross entropy**  
	分类交叉熵
- ...

### 模型训练 Model Training
模型训练的目的，是寻找一组最优的**模型参数 Parameter**，使损失最小。

训练一个模型的步骤：
- **Step 1**: 首先，定义一个损失函数。
- **Step 2**: 然后，最小化这个损失函数。
- **Step 3**: 从而获得最优的模型参数。

**Train a model ⟺ Minimize a function**

### 回归中的损失函数：Mean Squared Error (MSE)

$$
MSE = \frac{1}{N} \sum^N_{i=1}(y^i-\hat{y}^i)^2
$$

其中 $y^i$ 为实际值，$\hat{y}^i$ 为模型预测值。

- MSE 总是**凸 Convex 且可导 Differentiable** → 便于优化。
- MSE **对离群点** **outliers 不鲁棒 robust**。

离群点：与其他数据点明显不同的数据点。
![](Assets/Pasted%20image%2020251205103519.png)
- 如果第 $k$ 个样本是离群点，那么 $(y^i-\hat{y}^i)$​ 的差值会很大。
- 由于在 MSE 中该差值 $(y^i-\hat{y}^i)$ 被**平方**，**整体损失会被这些离群点主导**。

### 回归中的损失函数：Mean Absolute Error (MAE)

$$
MAE = \frac{1}{N} \sum^N_{i=1} |y^i-\hat{y}^i|
$$

- MAE 对离群点更加鲁棒。
- 但是它**不可导**，因此优化起来比较困难。

MSE 会放大（平方级）大误差的影响。为了最小化总损失，模型会被迫“用力”去迁就这个离群点，导致模型参数发生剧烈偏移，从而牺牲了对正常点的预测准确度。而 MAE 只是线性地看待这个错误，不会因为它是离群点就给予过度关注。

从统计学角度来说，MSE 等于估计数据的均值，MAE 等于估计数据的中位数。中位数受到个别离群点的干扰明显小于均值。

### 分类中的损失函数
#### 二元交叉熵 Binary Cross Entropy
二元交叉熵通常使用在分类数为2的情况。

$$
l = - \frac{1}{N} \sum^N_{i=1} y^i \log \hat{y}^i + (1-y^i) \log (1-\hat{y}^i)
$$

其中 $N$ 为样本数量。

#### 多类交叉熵 Categorical Cross Entropy
对于多类别分类问题，使用多类交叉熵。

$$
l = - \frac{1}{N} \sum^N_{i=1} \sum^K_{j=1} y^i_j \log \hat{y}^i_j
$$

其中 $K$ 为类别数量。

- 交叉熵是**可导的**（更易于优化）。
- 然而，它可能具有多个**局部极小值 Local Minima**。

### 线性回归模型的训练
在线性回归中，损失函数通常用 MSE 来定义：

$$
l(a,b)=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
$$

- $y^j$ 为真实值；
- $\hat{y}^j = \bf{ax^j} + b$ 为预测值。
代入可得：

$$
l(\pmb a,b)=\frac{1}{N}\sum_{i=1}^{N}(y_i- \pmb{ax}^j - b)^2
$$

#### 最小化损失
我们的目标是最小化损失，也就是求解下面这个无约束优化问题：

$$
\min_{a,b} l(a,b) = \min_{a,b} \frac{1}{N} \sum_{j=1}^{N} (y^j - a x^j - b)^2
$$

其中 $a, b$ 为需要被优化的决策变量。
- $l(a,b)$ 连**续可导且凸**。
- **梯度下降 Gradient Descent Algorithm** 适合用于最小化这个损失。

### 最小化损失函数：梯度下降 The Gradient Descent Method
梯度下降是一个一阶的迭代优化算法，用来寻找可微函数的局部最小值。
**梯度下降迭代 Gradient Descent Iteration：**

$$
\pmb{x}^{k+1} = \pmb{x}^k - \alpha^k \pmb \nabla \pmb l (\pmb x^k)
$$

其中 $\nabla$ 为nabla算子，$\alpha^k \gt 0$ 则是学习的步长，在 ML 中也称为 **学习率 Learning Rate**。
$\pmb{ \nabla l (x^k)}$ 是函数 $l$ 在点 $\pmb{x}^k$ 的梯度，即：

$$
\pmb{\nabla l(x^k)} = [\frac{\partial l}{\partial x^k_1},\frac{\partial l}{\partial x^k_2}, \dots, \frac{\partial l}{\partial x^k_n},]^T
$$

其中 $k$ 是当前迭代的数量，$n$ 则是向量 $\pmb{x}$ 的维度。

梯度下降算法步骤：
1. 将 $\pmb x^0$ 设置为任意初始点；
2. 按照式 $\pmb{x}^{k+1} = \pmb{x}^k - \alpha^k \pmb \nabla \pmb l (\pmb x^k)$ 进行迭代，直到满足 $|l(\pmb x^{k+1}) - l(\pmb x^k)| \le \epsilon$。

例如：使用梯度下降算法最小化：$l(x_1, x_2) = x_1^2 + x^2_2$ 
![](Assets/Pasted%20image%2020251205134900.png)

#### 批量梯度下降 Batch Gradient Descent (BGD)
对于一个线性回归，其损失函数可写为：

$$
l(\pmb a,b)=\frac{1}{N}\sum_{i=1}^{N}(y_i- \pmb{a \cdot x}^j - b)^2
$$

计算梯度：

$$
\frac{\partial l}{\partial a_i} = \frac{2}{N} \sum^N_{j=1} (\underbrace{\pmb{a \cdot x}^j}_{第j样本的个特征向量\pmb x^j的和权重向量点积} +b -y^j)\underbrace{x^j_i}_{第i个特征，常量}
$$

$$
\frac{\partial l}{\partial b} = \frac{2}{N} \sum^N_{j=1} (\pmb{a \cdot x}^j +b -y^j)
$$


- 批量梯度下降（BGD）每次使用全部 $N$ 个训练样本来计算梯度； 
- 然后使用平均梯度来更新模型参数（例如线性回归中的 $a$ 和 $b$）。

- **优点：在合适条件下收敛性有保证。**
- **缺点：探索能力较弱、内存消耗较大。**

#### 随机梯度下降 Stochastic Gradient Descent (SGD)
1. 从训练集中取出一个样本；
2. 计算该样本对应的梯度；
3. 按照梯度更新模型；
4. 对训练集中的所有样本重复步骤 1-4。

**优点：**
- **需要的内存更少。**
- **探索能力更强。**

**缺点：**
- **模型更新频繁，计算开销大；**
- **梯度噪声较大，收敛速度可能较慢。**

#### 小批量梯度下降 Mini-batch Gradient Descent (MBGD)
- 在 BGD 中，每次使用全部样本。
- 在 SGD 中，每次只使用一个样本。
- 在 MBGD 中，每次使用一个**小批量 Mini-batch。

一个小批量包含 $M$ 个训练样本，其中 $M \lt N$。
$M$ 称为批大小（batch size）。

1. 把数据集划分为若干个小批量；
2. 从训练集中取出一个小批量；
3. 计算该小批量样本的平均梯度；
4. 根据平均梯度更新模型；
5. 对所有小批量重复步骤 2–4。


在 MBGD 中，$M$ 是影响学习过程的一个**超参数 Hyperparameter**。
- **$M$ 较小时，行为类似于 SGD；**
- **$M$ 较大时，行为类似于 BGD。**

一个常用的默认Batch Size是 32。

MBGD 是 BGD 和 SGD 的一个结合，故
- **在一定程度上可以避免收敛到不好的局部最小点；**
- **所需内存较少；**
- **收敛速度较快。**

---
## ML 模型评估中的基础概念
### 训练数据与测试数据 Training & Test Data
在机器学习中，数据集通常被划分为**训练集和测试集**。
训练集用于训练机器学习模型，而测试集用于作为基准来评估一个已训练好的模型。
我们使用 **预测误差Prediction Error** 来评价模型：

**prediction error = Bias² + Variance + Noise**  
即 预测误差 = 偏差² + 方差 + 噪声。

**噪声 Noise**
- 噪声是模型也无法消除的**误差 error**，是**不可约 irreducible**的。

**偏差 Bias**
- 偏差是模型平均预测值与我们想要预测的真实值之间的差异。
- 它既来源于模型本身的假设，也来源于训练过程。

**方差 Variance**
- 方差是针对同一个数据点，模型预测结果的**变化程度 Variability**。
- 它与训练过程密切相关。


#### 欠拟合 Underfitting
偏差较大的模型，会在训练集和测试集上都产生较大的误差。
高偏差模型可能由于：
- 对训练数据做出了过于强烈的假设；
- **欠拟合 Underfitting**。

**欠拟合指的是：模型既不能很好地拟合训练数据，也不能很好地泛化到新的数据。**
**Underfitting refers to a model that can neither model the training data well nor generalize to new data.**
![](Assets/Pasted%20image%2020251205143305.png)
**检测欠拟合**
欠拟合通常比较明显，因为它在训练集和测试集上的表现都不好。

**避免欠拟合的方法：**
- 换用其他机器学习算法；
- 延长训练时间（多训练一些轮次）。

#### 过拟合 Overfitting
方差描述的是：当使用**不同的训练数据集**时，目标函数估计值改变的程度。
方差较高的模型，会在测试数据上产生较大的误差。
高方差往往意味着算法把训练数据中的**随机噪声也当成了模式来学习**，即发生了**过拟合 Overfitting**。

过拟合是指模型对训练数据“拟合得太好”，甚至把噪声也当成了有用的模式来学习。
![](Assets/Pasted%20image%2020251205143557.png)
**检测过拟合**
如果模型在训练集上的预测误差远远小于在测试集上的误差，就很可能已经过拟合。

**避免过拟合的方法：**
- 增加训练数据量；
- 移除与任务无关的特征；
- 使用**早停 Early Stopping**（在验证集上性能不再提升时停止训练）。

### $K$ 折交叉验证 $K$ Fold Cross-validation
$K$ 折交叉验证用于评价模型性能，并客观比较不同的模型或超参数设置。
数据集被划分为 $K$ 份，每次用其中 1 份作测试集，其余 $K−1$ 份作训练集，循环 $K$ 次，得到 $K$ 个评价分数，然后求**平均**等。
![](Assets/Pasted%20image%2020251205143909.png)
#### 通过 $K$ 折交叉验证来估计偏差和方差
在完成 $K$ 折交叉验证后，我们得到 $k$ 个预测误差估计值，即 $e_1, e_2, \dots e_k$；
**偏差可以通过求这些误差的均值估计：**

$$
\bar{e} = \frac{1}{k} \sum^k_{i=1} e_i
$$

。
**方差可以通过求这些误差的标准差估计：**

$$
\sigma = \sqrt{\frac{1}{k} \sum^k_{i=1} (e_i - \bar e)^2}
$$

。
基于误差的均值和方差，我们就可以对模型做出较客观的判断，也可以在此基础上调节超参数、比较不同的机器学习模型。

### 偏差-方差权衡 Bias-Variance Trade-off
偏差与方差之间存在此消彼长的关系：
- **增大偏差通常会减小方差；**
- **增大方差通常会减小偏差。**

所以，需要在偏差和方差之间取得一个好的平衡，从而使**总误差最小**。
最佳情况下**既不过拟合也不过欠拟合**。
![](Assets/Pasted%20image%2020251205150548.png)

### 泛化 Generalization
**泛化**指的是：机器学习模型从训练数据中学到的知识，在未见过的新数据上表现得有多好。

一个好的机器学习模型，应该能够把从训练数据中学到的模式很好地泛化到该问题域中的其他数据上。
