# 高级数据预处理与特征选择 Advanced Data Processing & Feature Selection
## 高级数据预处理：降维 Dimension Reduction
- **定义**: 利用数学或统计方法将高维数据转换为低维表示 。
- **目的**: 尽可能保留原始数据关键信息，降低复杂性，提高效率 。
- **与特征选择的区别**: 特征选择直接丢弃特征，如果所有特征都重要会导致性能下降，因此需要降维（特征组合/转换） 。

- **缓解“维数灾难”**: 增加数据密度，提高泛化能力 。
	- **现象**: 随着维数增加，样本变得更容易区分，但由于样本数量没有增加，数据变得稀疏，导致模型容易过拟合 。
- **提高计算效率**: 加速训练，优化算法在低维空间更高效 。
- **去除噪声和冗余**: 移除无关特征，合并相关特征 。
- **改善可视化**: 将数据降至 2D/3D 以便观察 。
- **增强可解释性**: 如 PCA 主成分可解释为“综合指标” 。


- **协方差矩阵 (Covariance Matrix)**: 描述变量间的相关性。
- **特征分解 (Eigen-decomposition)**: $A=Q\Lambda Q^{T}$，用于对称矩阵。
- **奇异值分解 (Singular Value Decomposition, SVD)**: $A=U\Sigma V^{T}$，适用于任意 $m \times n$ 矩阵 。

### **降维的评估**
**保留方差比例（Retained variance ratio）**

$$
\frac{\operatorname{Var}(X)}{\operatorname{Var}(X_r)}
$$

- $X$：原始数据  
- $X_r$：降维后的数据  

**重构误差（Reconstruction Error）**

$$
Z = \operatorname{Encoder}(X)
$$

$$
\hat{X} = \operatorname{Decoder}(Z)
$$

$$
\operatorname{Error}(X, \hat{X})
$$

- $X$：原始数据  
- $\hat{X}$：重构后的数据  
- $\operatorname{Error}$：误差度量，如 MSE、MAE 等  

**下游任务表现（Downstream task performance）**
例如：使用降维后的数据进行识别任务时的分类/识别准确率。

### 潜在的误解 potential misconceptions
1. **“降维一定会提升模型性能？”**  
   降维有时只会提升计算效率，反而可能使预测准确率下降。

2. **“降维后保留的主成分都可以被直接解释？”**  
   通过降维方法保留下来的某些特征，在语义上可能难以解释。

### 线性与非线性降维 Linear and non-linear dimensionality reduction

### 线性降维
通过寻找一个线性变换（原特征空间的线性组合），将数据映射到一个低维子空间中。  
常见方法包括：主成分分析（PCA）、线性判别分析（LDA）、奇异值分解（SVD）等。

### 非线性降维
利用非线性变换对数据进行建模或映射，从而在低维空间中获得数据表示。  
常见方法包括：核 PCA、局部线性嵌入（LLE）、t 分布随机邻域嵌入（t-SNE）、变分自编码器（VAE）等。

## 线性降维：主成分分析 Principal Component Analysis

### **PCA 原理**
![](Assets/Pasted%20image%2020251211105641.png)
- **核心思想**: 计算各维度的方差并排序。方差小的维度变化小，信息量少 。
- **操作**: 选择排名靠前的维度作为主成分，移除排名低的 。

### PCA 操作
$X$: 一个 $n \times p$ 的矩阵。其中 $n$ 为样本数量， $p$ 为特征数量。
#### Step 1 归一化
![](Assets/Pasted%20image%2020251211105809.png)
- **公式**: $X_{centered} = X - \mu$ 。
- **目的**: 消除不同特征量纲和偏差的影响，使 PCA 真正捕捉方差而非均值。

#### Step 2 计算协方差矩阵
- **公式**:

$$
\Sigma = \frac{1}{n-1}X_{centered}^{T}X_{centered}
$$

- **含义**: 对角线是方差，非对角线是协方差。

协方差矩阵 $\Sigma$ 是一个 $p \times p$ 的对称矩阵，其中对角线元素是各个特征的方差，非对角线元素是特征之间的协方差。  
由于在数据归一化时，所有维度的均值已经被调整为 0，因此可以直接使用上述形式来计算协方差矩阵。

#### Step 3 特征分解 Conducting Eigen-Decomposition
对协方差矩阵做特征分解：

$$
\Sigma = Q \Lambda Q^{T}
$$

- $Q$ 是一个 $p \times p$ 的矩阵，每一列 $q_i$ 是一个特征向量（即一个主成分的方向）；  
- $\Lambda$ 是对角矩阵，对角线上的元素 $\lambda_1, \lambda_2, \ldots, \lambda_p$ 为对应的特征值。  
  特征值 $\lambda_i$ 的大小表示该主成分所携带的方差信息量。

**如何得到 $Q$ 和 $\Lambda$？**
通过求解特征方程：

$$
\det(\Sigma - \lambda I) = 0
$$

得到各个特征值 $\lambda_i$，其中 $I$ 为单位矩阵。对于每个特征值 $\lambda_i$，求解

$$
(\Sigma - \lambda_i I) v_i = 0
$$

可以得到对应的特征向量 $v_i$。

#### Step 4 选择主成分
将所有特征值及其对应的特征向量按照特征值从大到小排序。  
特征值 $\lambda_i$ 表示在该主成分方向上的方差大小。

为每个主成分计算其解释方差比（Explained Variance Ratio，EVR）：

$$
\text{EVR}_j = \frac{\lambda_j}{\sum_{i=1}^{p} \lambda_i}
$$

根据需求选择前 $k$ 个主成分。

**典型的 $k$ 选择方法：**
- 直接指定一个固定的降维维度，将其设为 $k$；  
- 指定一个“原始方差保留率”，选取前 $k$ 个主成分，使得累计 EVR 达到该保留率；  
- 绘制特征值大小的折线图（scree plot），在“拐点”处选择合适的 $k$。

#### Step 5：数据映射（降维） Mapping the data, dimensionality reduction
利用选取出的特征向量构造映射矩阵 $W_k$（一个 $p \times k$ 的矩阵），然后计算：

$$
Z = X_{\text{centered}} W_k
$$

其中，$Z$ 是一个 $n \times k$ 的矩阵，即完成降维后的数据表示。

### 评价
**优点**
- **去噪（Denoising）：** 较小的主成分往往对应噪声，去除这些主成分可以提高信噪比。  
- **避免冗余（Avoiding redundancy）：** 各主成分方向两两正交，意味着新特征之间完全不相关，减少了冗余信息。

**缺点**
- **可解释性降低（Decreased interpretability）：** 主成分是原始特征的线性组合，其物理含义可能变得不直观、不易解释。  
- **依赖方差（Dependence on variance）：** 该方法倾向于保留方差较大的特征方向。如果存在方差较小但非常重要的特征，它们可能在降维过程中被舍弃。

## 线性降维：线性判别分析 Linear Discriminant Analysis
找到一条直线，使得所有数据点投影到这条直线上后满足：
![](Assets/Pasted%20image%2020251211142711.png)
- **最大化类间间隔（Maximize inter-class separation）**：不同类别数据在该直线上的投影尽可能远。  
- **最小化类内方差（Minimize intra-class variance）**：同一类别的数据在该直线上的投影尽可能聚得很紧。

LDA 的目标就是寻找这样一个**最优投影方向**，在该方向上线性投影后，不同类别的数据尽可能分开。

#### Step 1 计算各类的均值向量（Calculating the means）

设 $X$ 是一个 $n \times p$ 的数据矩阵，$n$ 为样本数，$p$ 为特征数。  

为每一类计算其均值向量 $\mathbf m_i$，并计算所有数据的整体均值向量 $\mathbf m$。

#### Step 2：计算散度矩阵（Calculating the scatter matrices）

**类内散度矩阵（Within-class scatter matrix）：**

$$
S_W = \sum_{i=1}^{c} S_i
= \sum_{i=1}^{c} \sum_{x \in D_i} (x - \mathbf m_i)(x - \mathbf m_i)^{T}
$$

**类间散度矩阵（Between-class scatter matrix）：**

$$
S_B = \sum_{i=1}^{c} N_i (\mathbf m_i - \mathbf m)(\mathbf m_i - \mathbf m)^{T}
$$

其中：

- $c$：类别数（number of categories）  
- $D_i$：第 $i$ 个类别中的样本集合  
- $N_i$：第 $i$ 个类别的样本数量  


#### Step 3 构造目标矩阵（Calculating the objective）

$$
S = S_W^{-1} S_B
$$

#### Step 4：特征分解（Eigen-decomposition）

对矩阵 $S$ 做特征分解，得到特征值 $\lambda_i$ 及其对应的特征向量 $\mathbf q_i$。

#### Step 5 选择投影方向（Select the projection direction）

将特征值按从大到小排序，选取前 $k$ 个最大特征值对应的特征向量构成投影矩阵 $W$（一个 $p \times k$ 的矩阵），其中 $k \le c - 1$。

#### Step 6 数据映射（降维）（Mapping the data, dimensionality reduction）

$$
Z = X W_k
$$

其中，$Z$ 为一个 $n \times k$ 的矩阵，即降维后的数据表示。


#### Step 7（选）：在新数据上进行分类（Classifying based on new data）

一种典型方法：

- 计算每一类在投影空间中的中心点 $\mathbf m_i'$；  
- 对于一个新样本 $x_{\text{new}}$，先进行投影：
  $$
  z_{\text{new}} = x_{\text{new}}^{T} W
  $$
- 使用分类器（例如最近中心分类器），预测其类别：
  $$
  \arg\min_{i} \left\| z_{\text{new}} - \mathbf m_i' \right\|^{2}
  $$

**Advantages（优点）**
- **有监督学习（Supervised）：** 在降维过程中显式利用类别标签信息，相比无监督方法，通常能在分类任务上获得更好的降维效果。  
- **可解释性较强（Interpretability）：** 投影方向（特征向量）代表了最具判别力的特征组合。

**Disadvantages（缺点）**
- **依赖高斯假设（Reliance on the Gaussian assumption）：** LDA 的推导隐含假设数据服从高斯分布（正态分布），在真实数据中这一假设未必成立。  
- **对离群点敏感（Sensitivity to outliers）：** 均值和散度矩阵的计算很容易受到异常点的影响。


## PCA 与 LDA 的区别

| Aspect（方面）            | PCA                                                      | LDA                                                                 |
|---------------------------|----------------------------------------------------------|----------------------------------------------------------------------|
| Learning Type（学习类型） | 无监督学习（Unsupervised learning，标签不需要）          | 有监督学习（Supervised learning，需要标签）                         |
| Optimization Objective（优化目标） | 最大化数据方差，尽可能保留“总体信息量”                  | 最大化类间可分性，使不同类别更加可分（inter-class discrimination） |
| Dimensionality Reduction Limit（降维维度上限） | 可以降到任意维度（理论上可到 1 维）                       | 最高只能降到 $c-1$ 维（$c$ 为类别数）                               |
| Application Focus（应用侧重点） | 通用降维、去噪与可视化                                   | 用于特征提取和面向分类任务的降维                                   |
| Effect（效果）            | 更清晰地展示数据的全局结构                               | 更强的类别分离效果，增强类间区分                                   |


## 线性降维：奇异值分解 Singular Value Decomposition, SVD
任意矩阵都可以看作是对数据的一种“视角”（例如原始特征维度）。  
SVD 的目标是找到一组新的、更本质的正交基，从这些新视角重新描述数据，从而揭示其内部结构。
![](Assets/Pasted%20image%2020251211143348.png)


#### Step 1 归一化（Normalization）

设 $X$ 为一个 $n \times p$ 的矩阵，其中 $n$ 为样本数、$p$ 为特征数。

$$
A = X - \mu
$$

其中 $\mu$ 为各维度的均值向量。

目的：消除不同特征维度与偏置的影响，使 SVD 捕捉到的主要是“方差结构”而不是均值偏移。

#### Step 2 进行奇异值分解

$$
A = U \Sigma V^{T}
$$

- $U$：$n \times n$ 的正交矩阵（orthonormal matrix）；  
- $V$：$p \times p$ 的正交矩阵；  
- $\Sigma$：对角矩阵（diagonal matrix），对角线元素为矩阵 $A$ 的奇异值（singular values），可能存在全零的行或列。

一种实现方式：

- 计算 $A^{T}A$ 并对其做特征分解；  
- 得到的特征向量可用于构造 $U$ 和 $V$，对应的特征值的平方根即为 $A$ 的奇异值。

#### Step 3 选择目标维度数 Selecting a target dimension number 

计算**前 $k$ 个奇异值的平方和与全部奇异值平方和的比例**。  
奇异值平方 $\sigma_i^{2}$ 是协方差矩阵 $A^{T}A$ 的特征值，表示对应方向的方差大小。

对第 $j$ 个成分，其解释方差比（Explained Variance Ratio, EVR）为

$$
\text{EVR}_{j}
= \frac{\sum_{i=1}^{j} \sigma_i^{2}}
       {\sum_{i=1}^{\min(n,p)} \sigma_i^{2}}
$$

通常选择最小的 $k$，使得累计解释方差率达到某个阈值（如 95% 或 99%）。

另一种做法是使用 scree plot：  
绘制奇异值随排序的下降曲线，在“拐点”处选择合适的 $k$。

#### Step 4 数据映射（降维）

**方法 1：投影到右奇异向量上（在特征空间中降维）**

$$
Z = A V_k = (U_k \Sigma_k V_k^{T}) V_k = U_k \Sigma_k
$$

其中 $V_k$ 为 $V$ 的前 $k$ 列。  
这是最常用的方法。$Z$ 的每一行是一个样本在新的 $k$ 维特征空间中的坐标，新特征通常是原始特征的线性组合。

**方法 2：直接使用左奇异向量和奇异值**

$$
Z = U_k \Sigma_k
$$

该结果与方法 1 完全等价，$Z$ 即为降维后的数据。

### SVD 的优缺点

**Advantages（优点）**

- **数值稳定性强（Numerical stability）**：是求解线性代数问题最稳定、最可靠的方法之一。  
- **适用性广（Strong universality）**：适用于任意矩阵，不要求矩阵是方阵或满秩。  
- **最佳近似性质（Best approximation）**：Eckart–Young–Mirsky 定理保证，截断 SVD 在 Frobenius 范数意义下给出了原矩阵的最佳低秩近似。

**Disadvantages（缺点）**

- **计算成本高（High computational cost）**：对大矩阵做完整 SVD 的计算代价很高，不过可以使用随机/截断 SVD 算法降低开销。  
- **可解释性较弱（Low interpretability）**：新的潜在特征通常是原始特征的线性组合，其物理含义可能不直观。  
- **线性假设（Linear assumption）**：只能捕捉数据中的线性结构和模式，对强非线性结构的表达能力有限。


### PCA 与 SVD 的区别与联系
- **演示**:
    
    - 未中心化的数据直接做 SVD 会受均值影响 。
    - PCA 先中心化再分解，真正找到方差方向 。

- **联系**: 对**中心化**的数据矩阵 A 进行 SVD，其右奇异矩阵 V 就是 PCA 的主成分方向 。

- **主要区别**: SVD 可用于任意矩阵，PCA 只能用于协方差矩阵 。
