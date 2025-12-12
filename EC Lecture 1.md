#### 优化 Optimization

常见的**优化问题 Optimization problem**：  
**旅行商问题 Traveling Salesman Problem (TSP)** /  
**二次指派问题 Quadratic Assignment Problem (QAP)** /  
**背包问题 Knapsack problem**

---

#### 目标与约束 Objectives & Constraints

例如：

$$
\min\{ f(x) \} = - (x-1)^2 \quad \text{for } -3 \le x \le 3
$$

这是**目标函数 Objective function**。

$$
s.t.\ g(x) = x - 1.5 \ge 0
$$

这是**约束 Constraint**。

- **可行域 Feasible space**：满足所有约束条件的所有可能解的集合。  
- **可行解 Feasible solution**：位于可行域中的一个具体解。

一个**优化问题 Optimization problem**的三个基本要素：  
**决策变量 Decision variables、目标函数 Objective functions 与约束 Constraints**。

- 对于**最小化问题 Minimization problem**：目标值越小越好；（若是最大化问题则相反）
- 对于**约束 Constraint**：只区分“可行”或“不可行”：
  - 所有可行解在可行性上是等价的。
  - 不可行解之间可以根据**违约程度 Violation**判断好坏。

---

#### 多目标 Multi-objectives

##### 支配 Domination

若有两个解 $x$ 与 $y$，若满足：

- 在所有**目标 Objective**上，解 $x$ 都不比解 $y$ 差；  
- 在至少一个目标上，解 $x$ 比解 $y$ 严格更优；

则称：

- $x$ **支配 dominates** $y$。

对于任意两个解 $x, y$，如果一方支配另一方，则它们不再被视为“同等好”的解。

##### 帕累托最优解 Pareto-Optimal Solution

- **帕累托最优解 Pareto-optimal solution**：指不被任何其他解支配的解。  
- **帕累托解集 Pareto Set, PS**：所有帕累托最优解的集合。  
- **帕累托前沿 Pareto-optimal front, PF**：所有帕累托最优解在**目标空间 Objective space**中的函数值集合。  
- 数学上：$\text{PF} = F(\text{PS})$。

##### 解的优先级 Solution ranks

- 可行解 Feasible solution 优于 不可行解 Infeasible solution。  
- 两个可行解：由支配关系决定，**支配者 Dominating solution** 更好。  
- 两个不可行解：违背约束越少、违背程度越小者更好。

---

#### 全局与局部优化 Global & Local Optimization

##### 全局最优解 Global optimal solution

考虑如下**最大化问题 Maximization problem**：

$$
\max_{x \in \Omega} f(x)
$$

若存在 $x^* \in \Omega$，满足：

$$
f(x) \le f(x^*) \quad \forall x \in \Omega
$$

则称 $x^*$ 为**全局最大解 Global maximal solution**。

> 一个优化问题可能存在多个**全局最优解 Global optimal solutions**。

##### 邻域 Neighborhood

- 定义：与某一点 $x$“足够接近 close”的一组点。  
- “接近 close”的具体定义依赖于  
  1. 具体问题本身；  
  2. 自己选择的**距离度量 Distance measure**或邻域结构。

##### 局部最优解 Local optimal solution

- 定义：在其**邻域 Neighborhood**内不比其他任一点更差的点。  
- 是否为局部最优解依赖于邻域的定义。  
- 性质：**全局最优解 Global optimal solution 一定是局部最优解 Local optimal solution**。

##### 局部优化 Local optimization

- **局部优化 Local optimization**旨在找到某个局部最优点，或在**搜索空间 Search space**的某个特定区域内找到最优解。  
- 通常需要一个**初始点 Starting point**来说明从哪个区域开始搜索。

##### 全局优化 Global optimization

- **全局优化 Global optimization**尝试在整个搜索空间中找到一个或多个全局最优点。  
- 可能有初始点，也可能没有；多数情况下只给出搜索空间。  
- 一般**不保证 guarantee**一定能找到全局最优解，只是“尽量接近”。

##### 单峰与多峰优化问题 Unimodal and multimodal optimization problems

- **单峰 / 凸问题 Unimodal / Convex problem**：目标函数的“地形 landscape”只有一个最优值（没有多个局部极小）。  
- **多峰问题 Multimodal problem**：目标函数存在多个局部极小值 **Local minima**。


