#### Optimization
Popular optimization problem: TSP / QAP / knapsack
#### Objectives & Constraints
e.g.
$\min\{ f(x) \}  = - (x-1)^2 \text{ for } -3 \leq x \leq 3$ (Objective Function)
$s.t.\ g(x) = x -1.5 \geq 0$
(Constraint)

- Feasible Space: the feasible space refers to the set of all possible solutions that satisfy a given set of constraints.
- Feasible Solution: a solution in the feasible space.

Three elements of an optimization problem: **decision variables, objective functions and constraints.**
- **Objective**: the smaller the better (considering a minimization problem, how about maximization?) 
- **Constraint**: only feasible or infeasible:
	- feasible solutions are considered the same. 
	- infeasible solutions have different quality.

#### Multi-objectives
##### Domination(支配)
if
- 解 $x$ 在任何objective都不比解 $y$ 差
- 解 $x$ 在至少一个objective比解 $y$ 严格更优
$x$ dominates / 支配 $y$

对于任意两个解x，y，如果x和y其中一方支配另一方，就不可以彼此compare。

##### Pareto-Optimal Solution (Pareto 最优解)

- **Pareto-Optimal Solution** 即一个不被其他任何解支配的解
- **Pareto Set (PS)** 所有Pareto最优解的集合
- **Pareto-Optimal Front (PF)** 所有Pareto最优解的Objective Function
- PF = F(PS)

##### Solution Ranks
- Feasible Solution > Infeasible Solution
- Two Feasible Solution: The one that dominates
- Two Infeasible Solution: The one that violates less

#### Global & Local Optimization
##### Global Optimal Solution
优化问题：
$$
\max_{x \in \Omega}
$$
若$\exists x^* \in \Omega$，满足：
$$
f(x) \leq f(x^*) \ \ \forall x \in \Omega
$$
则$x^*$是Global Maximal Solution。

**一个优化问题可能有多个全局最优解**

##### Neighborhood (邻域)
- Definition: A number of points which are **close** to x.
- ***close的定义取决于 (1) 问题  (2)自己的选择*

##### Local Optimal Solution
- Def: A point that is not worse than any other points in its neighborhood.
- Depends on the neighborhood.
- **A global optimal solution is always local optimal.**

##### Local Optimization
- Local optimization aims to find a local optimal point or find the optimal solution from **a specific region** of the search space.
- **It often has a starting point to define where is the specific region.**

##### Global Optimization
Global optimization attempts to find (a) global optimal point(s) of the search space.
- It may have a starting point or not. In many cases, only the search space is provided.
- It does not guarantee to find (a) global optimal point(s).

##### Unimodal and multimodal (单峰和多峰) optimization problems
- **Unimodal / Convex**: the landscape has a single optimal value.
- **Multimodal**: the landscape has more than one local minimum.