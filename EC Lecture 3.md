# 受约束优化 Constrained optimization

## 目标函数 vs. 约束 Objective vs. Constraint

示例问题：

$$
\min f(x) = (x-1)^2 - 1
$$

$$
\text{s.t. } g(x) = x - 1.5 \ge 0
$$

- **目标函数 Objective function**：需要被最小化/最大化的函数（这里是最小化）。
- **约束 Constraint**：可行解必须满足的条件（这里要求 $x \ge 1.5$）。

---

## 在受约束优化中比较解 Comparing solutions in constrained optimization

问题：

$$
\min f(x) = (x-1)^2 - 1,\quad \text{s.t. } g(x)=x-1.5 \ge 0
$$

例子（第 0 代/若干候选点）：

| $x$ | $f(x)$ | $g(x)$ |
|---:|---:|---:|
| 0   | 0     | -1.5 |
| 1   | -1    | -0.5 |
| 1.5 | -0.75 | 0 |
| 2   | 0     | 0.5 |

---

## 目标与约束的区别 Objectives and constraints

- **目标 Objective**：越小越好（针对最小化问题；若是最大化，则需做相应处理）。
- **约束 Constraint**：只有 **可行 Feasible** / **不可行 Infeasible** 两类  
  - 可行解通常被视为“同一层级”（都满足约束）。  
  - 不可行解之间仍可比较“好坏”（取决于违反程度）。

---

## 受约束优化的解排序 Rank solutions for a constrained optimization problem

如何排序？

- **可行解 Feasible solutions** 优于 **不可行解 Infeasible solutions**
- 两个可行解：**目标函数值 Objective value** 更优者更好（最小化时更小更好）
- 两个不可行解：**约束违反 Constraint violation** 更小者更好

---

# 处理约束：违反程度与惩罚函数 Handling constraints

## 描述约束违反 Handling constraints: describe constraint violation

### 约束违反之和 Sum of the constraint violation
- **Case 1**：总重量 $10\text{ kg}$ vs. 要求 $7\text{ kg}$（违反 $3\text{ kg}$）；  
  总体积 $70\text{ cm}^3$ vs. 要求 $60\text{ cm}^3$（违反 $10\text{ cm}^3$）
- **Case 2**：某些约束可能有 **优先级 Priority**

### 加权约束违反之和 Weighted sum of the constraint violation
- 当约束有优先级或量纲不同，使用 **加权和 Weighted sum** 来综合违反程度。

---

## 惩罚函数法 Penalty function method

将受约束问题：

$$
\min f(x)\quad \text{s.t. } g(x)\le c
$$

等价变换为无约束问题：

$$
\min \; f(x) + k \cdot h(g(x),c)
$$

其中 **惩罚函数 Penalty function**：

$$
h(g(x),c) = \max\{0,\; g(x)-c\}
$$

解释：
- $k$ 是 **惩罚系数 Penalty coefficient**（标量 scalar，例如 $k=50$），用于控制惩罚强度。
- 约束违反越多，惩罚项越大。
- 当满足约束时（$g(x)\le c$），惩罚项为 $0$，对目标函数无影响。

---

## 如何确定惩罚系数 $k$ How to determine the penalty coefficient $k$?

一种 **经验方法 Empirical method**：

- **Step 1**：估计一个可容忍的 **预期可接受违反 Intended acceptable violation** $\epsilon$  
  （现实问题中通常 $\epsilon>0$）
- **Step 2**：估计一个期望获得的 **期望最优目标值 Expected optimal objective value** $\tilde f$  
  （不一定准确，但可作为量级参考）
- **Step 3**：选择 $k$，使得 $k\times \epsilon$ 大约是 $\tilde f$ 的 $50$ 倍：

$$
k\cdot \epsilon = 50\tilde f
\quad\Rightarrow\quad
k = \frac{50\tilde f}{\epsilon}
$$

---

## 惩罚函数法：示例展开 Handling constraints: penalty function method (example)

![](Assets/Pasted%20image%2020251213154830.png)

给定：

$$
\min f(x)=(x-1)^2-1,\quad \text{s.t. } g(x)=x-1.5\ge 0
$$

若取 $k=500$，则等价的惩罚形式可写为（将违反写成 $\max\{0,1.5-x\}$）：

$$
f(x) + 500\max\{0,\;1.5-x\}
$$

也可写成分段形式：

$$
(x-1)^2-1+500\max\{0,1.5-x\}
=
\begin{cases}
(x-1)^2-1, & x-1.5\ge 0\\
(x-1)^2-1+500(1.5-x), & \text{otherwise}
\end{cases}
$$

---

## 多个约束怎么办？ What if there are multiple constraints?

原问题：

$$
\min f(x)\quad
\text{s.t. }
\begin{cases}
g_1(x)\ge c_1\\
g_2(x)\le c_2\\
g_3(x)\le c_3\\
\vdots
\end{cases}
$$

**Step 1：统一约束方向 Unify constraints**（例如全转成 $\le$）：

$$
\min f(x)\quad
\text{s.t. }
\begin{cases}
-g_1(x)\le -c_1\\
g_2(x)\le c_2\\
g_3(x)\le c_3\\
\vdots
\end{cases}
$$

**Step 2：用惩罚函数法并入目标 Integrate constraints**：

$$
\min\; f(x)
+ k_1 h(-g_1(x),-c_1)
+ k_2 h(g_2(x),c_2)
+ k_3 h(g_3(x),c_3)
+ \cdots
$$

其中 $k_i$ 是第 $i$ 个约束的 **惩罚强度 Penalty strength**。

**Step 3：分别确定 $k_i$ Determine each $k_i$**：

$$
k_i = \frac{50\tilde f}{\epsilon_i}
$$

$\epsilon_i$ 为针对第 $i$ 个约束的可接受违反。

---

## 如果是最大化？ What if we are doing maximization?

若：

$$
\max f(x)\quad
\text{s.t. }
\begin{cases}
g_1(x)\ge c_1\\
g_2(x)\le c_2\\
g_3(x)\le c_3\\
\vdots
\end{cases}
$$

可转换为最小化：

$$
\min -f(x)\quad
\text{s.t. }
\begin{cases}
-g_1(x)\le -c_1\\
g_2(x)\le c_2\\
g_3(x)\le c_3\\
\vdots
\end{cases}
$$

要点：只需对目标取负，把 **最大化 Maximization** 转成 **最小化 Minimization**，其余处理同前。

---

## 惩罚函数法是否遵循排序原则？ Does the penalty function method follow these principles?

排序原则（回顾）：
- 可行解优于不可行解
- 两个可行解比目标值
- 两个不可行解比违反程度

结论：**如果惩罚系数 $k$ 选择得当 Appropriate penalty coefficient**，惩罚函数法可以遵循上述原则。

---

# 例子：背包问题 Knapsack problem

## 问题设置 Problem setup

五个物品（价值 value / 重量 weight / 体积 volume）：

- 物品 1 item 1：$\$10$，$1\text{ kg}$，$10\text{ cm}^3$
- 物品 2 item 2：$\$25$，$3\text{ kg}$，$20\text{ cm}^3$
- 物品 3 item 3：$\$20$，$2\text{ kg}$，$25\text{ cm}^3$
- 物品 4 item 4：$\$15$，$2\text{ kg}$，$15\text{ cm}^3$
- 物品 5 item 5：$\$40$，$4\text{ kg}$，$30\text{ cm}^3$

背包容量 Capacity：$7\text{ kg}$ 且 $60\text{ cm}^3$

**决策变量 Decision variable**：

$$
\mathbf{X}=[x_1,x_2,\ldots,x_5],\quad
x_i=
\begin{cases}
1, & \text{选择 choose 物品 item } i\\
0, & \text{否则 otherwise}
\end{cases}
$$

最大化总价值：

$$
\max_{\mathbf{X}}\; 10x_1+25x_2+20x_3+15x_4+40x_5
$$

约束：

$$
x_1+3x_2+2x_3+2x_4+4x_5 \le 7
$$

$$
10x_1+20x_2+25x_3+15x_4+30x_5 \le 60
$$

---

## Step 1：转为最小化 Convert to minimization

$$
\min_{\mathbf{X}}\; -\left(10x_1+25x_2+20x_3+15x_4+40x_5\right)
$$

约束不变。

---

## Step 2：统一约束方向 Unify constraints

本例两个约束已经是 $\le$ 形式，因此 **无需额外转换 Already satisfied**。

---

## Step 3：惩罚函数并入目标 Integrate constraints by penalty method

$$
\min_{\mathbf{X}}\;
-\left(10x_1+25x_2+20x_3+15x_4+40x_5\right)
+ k_1\max\{x_1+3x_2+2x_3+2x_4+4x_5-7,\;0\}
+ k_2\max\{10x_1+20x_2+25x_3+15x_4+30x_5-60,\;0\}
$$


给定（示例选择）：

$$
\tilde f = \$100,\quad
\epsilon_1 = 1\text{ kg},\quad
\epsilon_2 = 10\text{ cm}^3
$$

则（对应 $k_i=\frac{50\tilde f}{\epsilon_i}$）：

$$
k_1=\frac{50\cdot 100}{1}=5000,\quad
k_2=\frac{50\cdot 100}{10}=500
$$

最终惩罚形式：

$$
\min_{\mathbf{X}}\;
-\left(10x_1+25x_2+20x_3+15x_4+40x_5\right)
+5000\max\{x_1+3x_2+2x_3+2x_4+4x_5-7,\;0\}
+500\max\{10x_1+20x_2+25x_3+15x_4+30x_5-60,\;0\}
$$


---
## 多目标优化问题 **Multiobjective optimization problem (MOP)**

$$
\begin{aligned}
\min f_1(x_1, x_2 ..., x_n) \\
\min f_2(x_1, x_2 ..., x_n) \\
\vdots \\
\min f_m(x_1, x_2 ..., x_n)
\end{aligned}
$$

$$
x = (x_1, x_2, ..., x_n)^T \in D \subseteq \mathbb{R}^n
$$

* $x$ 是 决策变量 **decision variables**，
* $n$ 是 $x$ 的 维度 **dimension**，
* $D$ 是 决策空间 **decision space**，
* $m$ 个 目标 **objectives** $f_1, f_2, ..., f_m$ 通常彼此冲突。
* 最大化 **Maximization** 的定义方式类似。

> **注意**：此后考虑的问题均为 最小化 **minimization** 问题。

---

## 示例：汽车设计 **Example: automobile design**

* 在设计汽车时，我们关注几个方面，如性能、安全性、能源效率和生产成本。

$$
\begin{aligned}
\min f_1(x_1, x_2 ..., x_n) \\
\min f_2(x_1, x_2 ..., x_n)
\end{aligned}
$$

$$
x = (x_1, x_2, ..., x_n)^T \in D \subseteq \mathbb{R}^n
$$

* 能源消耗（每百英里）应尽可能低。**越低越好 The lower, the better**。
* 制造成本应尽可能低。**越低越好 The lower, the better**。
* 它们彼此冲突。低能耗依赖于更先进的动力单元设计，这反过来会增加 制造费用 **cost of manufacturing**。

---

## 如何衡量解的质量？支配 **How to measure the quality of solutions? dominance**

### 哪些解是最优的？ **Which solutions are optimal?**

**支配 Dominance**：
$x$ 支配 **dominates** $y$ 如果：
* $x$ 在任何目标上都不比 $y$ 差 ($x \succcurlyeq y$)
* $x$ 至少在一个目标上严格优于 $y$ ($x \succ y$)

**备注 Remarks**：
对于任意两个解 $x, y$：
* $x$ 支配 **dominates** $y$，或
* $y$ 支配 **dominates** $x$，或
* $x, y$ 彼此无法比较。

---

## 哪些解是最优的？ **Which solutions are optimal?**

### 帕累托前沿 **Pareto front**：
在 $D$ 中不被任何解支配的点的集合。

*(图示展示了帕累托前沿曲线及被 $x$ 和 $y$ 支配的区域)*

![](Assets/Pasted%20image%2020251213162135.png)

---

## 帕累托前沿的不同形状 **Different shapes of the PF**

* 凸 **Convex**
* 凹 **Concave**
* 不连续 **Discontinuous**
* 既非凸也非凹 **Neither Convex nor Concave**

![](Assets/Pasted%20image%2020251213162146.png)

---

## 帕累托最优解 = 最佳权衡候选者 **Pareto optimal solutions = best trade-off candidates**

* $x$ 是帕累托最优的，当且仅当没有其他解支配它。
* 一个（理性的）决策者 **(rational) decision maker** 不会喜欢非帕累托最优解。

*(图示展示了从 搜索空间 **Search Space** (PS) 到 目标空间 (PF) 的映射)*

---

## 帕累托集与帕累托前沿 **Pareto set & front (PS & PF)**

**帕累托最优解 Pareto-optimal solution**：
不被任何其他解支配的解。

* **帕累托集 Pareto set (PS)**：所有 帕累托最优解 **Pareto optimal solutions** 的集合，是 决策空间 **decision space** 的子集。
* **帕累托前沿 Pareto front (PF)**：PS 中所有解的 目标函数值 **objective function values** 的集合，是 目标空间 **objective space** 的子集。

---

## 多目标优化问题 **Multiobjective optimization problem (MOP)**

$$
\begin{aligned}
\min f_1(x_1, x_2 ..., x_n) \\
\min f_2(x_1, x_2 ..., x_n) \\
\vdots \\
\min f_m(x_1, x_2 ..., x_n)
\end{aligned}
$$

$$
x = (x_1, x_2, ..., x_n)^T \in D \subseteq \mathbb{R}^n
$$

* $x$ 是 决策变量 **decision variables**，
* $n$ 是 $x$ 的 维度 **dimension**，
* $D$ 是 决策空间 **decision space**，
* $m$ 个 目标 **objectives** $f_1, f_2, ..., f_m$ 通常彼此冲突。
* 最大化 **Maximization** 的定义方式类似。

> **注意**：此后考虑的问题均为 最小化 **minimization** 问题。

---

## 示例：汽车设计 **Example: automobile design**

* 在设计汽车时，我们关注几个方面，如性能、安全性、能源效率和生产成本。

$$
\begin{aligned}
\min f_1(x_1, x_2 ..., x_n) \\
\min f_2(x_1, x_2 ..., x_n)
\end{aligned}
$$

$$
x = (x_1, x_2, ..., x_n)^T \in D \subseteq \mathbb{R}^n
$$

* 能源消耗（每百英里）应尽可能低。**越低越好 The lower, the better**。
* 制造成本应尽可能低。**越低越好 The lower, the better**。
* 它们彼此冲突。低能耗依赖于更先进的动力单元设计，这反过来会增加 制造费用 **cost of manufacturing**。

---

## 如何衡量解的质量？支配 **How to measure the quality of solutions? dominance**

### 哪些解是最优的？ **Which solutions are optimal?**

**支配 Dominance**：
$x$ 支配 **dominates** $y$ 如果：
* $x$ 在任何目标上都不比 $y$ 差 ($x \succcurlyeq y$)
* $x$ 至少在一个目标上严格优于 $y$ ($x \succ y$)

**备注 Remarks**：
对于任意两个解 $x, y$：
* $x$ 支配 **dominates** $y$，或
* $y$ 支配 **dominates** $x$，或
* $x, y$ 彼此无法比较。

---

## 支配：图解 **Dominance: illustration**

*(图示展示了 搜索空间 **Search space** 和 目标空间 **Objective space**)*

* $y$ 支配 **dominates** $x$ 当且仅当：
    * $y$ 在任何目标上都不比 $x$ 差，且
    * $y$ 至少在一个目标上优于 $x$。

在这个例子中：
* B 支配 **dominates** A；
* B 和 C 不可比较，A 和 C 也是如此。

---

## 哪些解是最优的？ **Which solutions are optimal?**

### 帕累托前沿 **Pareto front**：
在 $D$ 中不被任何解支配的点的集合。

*(图示展示了帕累托前沿曲线及被 $x$ 和 $y$ 支配的区域)*

---

## 帕累托前沿的不同形状 **Different shapes of the PF**

* 凸 **Convex**
* 凹 **Concave**
* 不连续 **Discontinuous**
* 既非凸也非凹 **Neither Convex nor Concave**

---

## 帕累托最优解 = 最佳权衡候选者 **Pareto optimal solutions = best trade-off candidates**

* $x$ 是帕累托最优的，当且仅当没有其他解支配它。
* 一个（理性的）决策者 **(rational) decision maker** 不会喜欢非帕累托最优解。

---

## 帕累托集与帕累托前沿 **Pareto set & front (PS & PF)**

**帕累托最优解 Pareto-optimal solution**：
不被任何其他解支配的解。

* **帕累托集 Pareto set (PS)**：所有 帕累托最优解 **Pareto optimal solutions** 的集合，是 决策空间 **decision space** 的子集。
* **帕累托前沿 Pareto front (PF)**：PS 中所有解的 目标函数值 **objective function values** 的集合，是 目标空间 **objective space** 的子集。

---

## 进化算法 **Evolutionary algorithms**

### 基于种群的迭代搜索方法 **Population-based iterative search methods**

简单 EA 的一次迭代过程：
1.  第 $t$ 代种群 **Population at generation t**
2.  选择 **Selection** -> 父代集合 **Parent Set**
3.  交叉/变异 **Crossover / Mutation**
4.  第 $t+1$ 代种群 **Population at generation t+1**

* **种群 Population**：候选解的集合。
* **选择 Selection**：选择适应度最高的解作为下一代的父代。
* **交叉 Crossover**：交换两个父代解的信息以产生新解。
* **变异 Mutation**：修改现有的解。

---

## 多目标进化算法 **Multiobjective evolutionary algorithms (MOEAs)**

生成 **一定数量的解 a number of solutions**：
* 以逼近 **approximate** 帕累托前沿 **PF (PS)**，或
* 为 决策者 **decision makers** 提供其他有用的信息，例如不同目标之间如何相互妥协。

* 在计算机科学和许多其他工程领域很流行。
* 多准则决策 **Multi-Criterion Decision Making** 中的主流方法。
* 进化计算领域最热门的研究领域（超过 15,000 篇研究论文）。

---

## 我们希望从 MOEA 中得到什么？ **What do we want from MOEA?**

有限数量的 **均匀分布 evenly distributed** 的帕累托最优解，能够很好地逼近 PF。

*(图示对比了两种分布：a) 分布均匀，逼近准确；b) 分布较差)*

直观地说，(a) 比 (b) 更准确地逼近了 PF。

---

## 给决策者的其他信息：示例 **Other information to the decision makers: an example**

假设我们尽最大努力找到了一打非支配解（如图所示为最小化问题），那么可能有用的信息包括：

* 目标 $f_1$ 可能比 $f_2$ 更容易优化；
* 沿着蓝色箭头指示的方向可能没有非支配解；
* 标记为绿色的解 $x^*$ 可能是一个“关键” **key** 解（拐点）。

---

## MOEA/D 的分类与优势

### MOEA/D: 基于分解的 MOEAs **Decomposition-based MOEAs**
* 将 MOP 分解为多个单目标子问题，并通过协同解决子问题来逼近 PF。

### NSGA-II: 基于支配的 MOEAs **Dominance-based MOEAs**
* 迭代地选择新种群，基于：1) 解之间的帕累托支配关系，2) 目标空间中解之间的距离。

### IBEA: 基于指标的 MOEAs **Indicator-based MOEAs**
* 应用一个指标（例如超体积 **hypervolume**）来评估一组解的质量，并根据解对指标的贡献迭代选择新种群。

### MOEA/D 的优势 **The advantages of MOEA/D**

在工程中：
* **小种群下具有良好的分布性和高优化质量 Good distribution & high optimization quality with a small-population**：
    * 由于仿真成本，种群规模通常限制在一个较小的数字；
    * 分解的思想有助于获得最终解的更均匀分布；
    * 考虑到小种群，MOEA/D 的性能优于 NSGA-II。

---

## MOEA/D: 分解 + 协同 **Decomposition + collaboration**

**分解 Decomposition (源自传统优化理论)**
* 将逼近 PF 的任务分解为 $N$ 个子任务。每个子问题通常是单目标的。

**协同 Collaboration (源自进化计算)**
* 使用 $N$ 个代理（过程）。每个代理针对一个不同的子问题。
* 这 $N$ 个子问题彼此相关。$N$ 个代理可以以协同的方式解决这些子问题。

### 子问题包含什么？ **What does a subproblem contain?**

输入 **Inputs**:
* 一个分解函数 **decomposition function** $g$
* 偏好向量 **preference vector** $\lambda$
* 可能包含一个参考点 **reference point** $z^*$
* $x$ 的 搜索空间 **search space** $D$
* 当前 最好解 **best-so-far solution**

目标：
$$\min g(x)$$

---

## MOEA/D 中的目标分解 **Objective decomposition in MOEA/D**

目标可以聚合为一个单目标函数（一个子问题）。
例如，可以用指定的权重 $\lambda_1$ 和 $\lambda_2$ 将它们求和：

$$g(x) = \lambda_1 f_1(x) + \lambda_2 f_2(x)$$
$$\text{subject to } \lambda_1 + \lambda_2 = 1$$

当用几种不同的权重对目标求和时，多目标问题就被分解成了几个不同的子问题。

**流行的分解方法 Popular decomposition methods**:
* 加权和法 **Weighted sum approach**
* 切比雪夫法 **Tchebycheff approach**

---

## 加权和法 **Weighted sum approach**

$$
\min g^{ws}(x | \lambda) = \lambda_1 f_1(x) + \lambda_2 f_2(x)
$$
$$
\text{where } \lambda_1 + \lambda_2 = 1 \text{ and } \lambda_1, \lambda_2 \ge 0.
$$

* 等高线 **Contour line**：同一等高线上的点具有相同的 $g$ 值。
* 它适用于 凸 PF **convex PF**。

*(图示展示了将 PF 逼近转化为 $K$ 个单目标优化子问题)*

---

## 切比雪夫法 **Tchebycheff approach**

对于任意帕累托最优解 $x^*$，存在一个 $\lambda$ 使得 $x^*$ 对于上述问题是最优的。

$$
\min g^{te}(x | \lambda, z^*)
$$
$$
g^{te}(x | \lambda, z^*) = \max \{ \lambda_1 | f_1(x) - z_1^* |, \lambda_2 | f_2(x) - z_2^* | \}
$$

* $z^* = (z_1^*, z_2^*)$ 是一个 理想点 **Utopian point**。
* $z_1^* < \min f_1$, $z_2^* < \min f_2$.

---

## 相邻子问题的协同 **Collaboration of neighboring subproblems**

**邻域 Neighborhood**：定义子问题之间的关系
* 如果两个子问题的 权重向量 **weight vectors** 相近，则它们互为邻居。
* 相邻的子问题应该具有相似的目标函数，因此大概率具有相似的最优解。

*(图示展示了问题 5 的邻域及其对应的权重向量)*

---

## 如何协同？ **How to collaborate?**

**一种简单的方法是在子问题及其邻居之间进行交叉 crossover between a subproblem and its neighbours**

* 子问题的当前最优解 (The best-so-far solution for the subproblem)
    * `[x1, x2, x3, x4, x5, x6, x7, x8, x9]`
* 从其邻居中随机选择的另一个当前最优解 (Another best-so-far solution randomly selected from one of its neighbours)
    * `[y1, y2, y3, y4, y5, y6, y7, y8, y9]`
* **交叉 Crossover** $\downarrow$
* 生成的后代解 (The reproduced offspring solution)
    * `[x1, x2, x3, y4, y5, y6, y7, x8, x9]`

---

## MOEA/D 的整体流程 **The overall procedure of MOEA/D**

在每一代，每个子问题执行以下操作：

1.  **交配池选择 Mating pool selection**：获取一些邻居的当前解（即 best-so-far）（**协同 collaboration**）。
2.  **繁殖 Reproduction**：通过应用 交叉 **crossover** 和 变异 **mutation** 生成一个新解。
3.  **替换 Replacement**：
    a) 如果新解在其目标方面更好，则替换其旧解。
    b) 将新解传递给其部分邻居，如果新解在它们各自的目标方面更好，则每个邻居都用该新解替换其旧解。（**协同 collaboration, 邻域 neighborhood**）


