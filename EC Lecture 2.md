# 遗传算法 Genetic Algorithm, GA

---
## GA 简介

- **种群 Population**  
  一群个体构成一个种群。  
  A number of individuals form a population.

- **选择 Selection：适者生存 Survival of the fittest**  
  个体越优秀，它被选作下一代父代的可能性就越大。  
  The better an individual is, the more likely it becomes a parent for the next generation.

- **交叉 Crossover**  
  子代的**染色体 Chromosome**由其父母的染色体组合而成。  
  The chromosome of a child is composed of chromosomes from its parents.

- **变异 Mutation**  
  在某些情况下，染色体的部分片段可以被随机改变。  
  In some occasions, some part of the chromosome can be randomly changed.


### 用“进化”求解优化问题 Use “evolution” to solve an optimization problem

- 在**搜索空间 Search space**中的一个**解 Solution** ↔ 自然进化中的一个**个体 Individual**。  
- 我们可以称之为解、个体或点 **Solution / Individual / Point**。  
- **进化 Evolution** 的目的：不断“创造 Create”更好的解。

#### 基本流程 Basic workflow

1. 从一开始的**解的种群 A population of solutions**出发。  
2. 通过 **选择 Selection** 得到**交配池 Mating pool（父代集合 Parent set）**。  
3. 在交配池中应用 **交叉 Crossover** 与 **变异 Mutation**，生成**新解 New solutions**。  
4. 通过 **替换 Replacement** 操作，用新解更新种群，得到新的解的种群。  
5. 重复以上步骤，逐步进化出更优的解。

---
## GA 表示，初始化，与选择 Representation, Initialization & Selection

### GA 工作流

![](Assets/Pasted%20image%2020251212225138.png)


1. **初始化 Initialization**  
   生成初始**种群 Population**：  
   $$
   \mathbf{x} = \{x^1, x^2, \dots, x^M\}
   $$
   其中 $M$ 为种群规模。

2. **选择 Selection**  
   从当前种群中选出**父代集合 Mating pool**：  
   $$
   \mathbf{x}_{\text{mat}} = \{x^1_{\text{mat}}, \dots, x^M_{\text{mat}}\}
   $$

3. **交叉 Crossover**  
   对父代进行交叉，产生新个体：  
   $$
   \mathbf{x}_{\text{cross}} = \{x^1_{\text{cross}}, \dots, x^M_{\text{cross}}\}
   $$

4. **变异 Mutation**  
   对交叉后个体进行随机变异：  
   $$
   \mathbf{x}_{\text{mut}} = \{x^1_{\text{mut}}, \dots, x^M_{\text{mut}}\}
   $$

5. **替换 Replacement**  
   用新个体替换部分或全部旧种群，形成下一代种群。

6. **终止条件 Termination conditions**  
   若满足终止条件（例如达到最大代数、目标函数不再改善等），  
   则**保存最优解 Save the optimal solution**；  
   否则返回步骤 2 继续迭代。


### 表示 Representation

- 一个**个体 Individual**使用某种数据结构进行编码。  

- 常见方式：**二进制串 Binary string** 编码，例如：

  ![](Assets/Pasted%20image%2020251212225231.png)

  每一位是一个**基因 Gene**。  
  通过 **解码 Decoding**，二进制串可以映射为一个实际数值（例如 $3$）；  
  通过 **编码 Encoding**，实际数值也可以转成二进制串。

- 其他表示方式 Other representations：  
  - **排列 Permutation**  
  - **实数 Real number**  
  - **整数 Integer** 等。


### 初始化 Initialization

- GA 从一个**初始种群 Initial population** 开始：  
  - **种群 Population** = 一组**个体 Individuals**。

- 需要考虑的两个关键问题：

  1. **种群规模 Population size**：需要多少个体？  
  2. **初始化方法 Initialization method**：如何生成初始种群？

- 初始种群通常可通过**随机采样 Random sampling**获得。

### 选择 Selection

选择策略模拟自然界中的**自然选择 Natural selection**与**适者生存 Survival of the fittest**。

- 决定哪些个体进入**交配池 Mating pool**。  
- 给“更好”的个体更多机会被选中 **Give more chances to better individuals**。  
- 通过**适应度函数 Fitness function**区分“较好 better”与“较差 poorer”的解。

常见的选择策略 Popular selection schemes：

- **比例选择 Proportional selection**  
- **锦标赛选择 Tournament selection**  
- **截断选择 Truncation selection**


#### 比例选择 Proportional selection

考虑最大化如下函数的优化问题：

$$
\max G(x) = -(x-8)^2 + 65,
$$

其中 $x$ 为整数，范围 $0 \le x \le 15$。

在比例选择中：

- 个体被选为父代的**概率 Probability**与其适应度成正比：

  $$
  \operatorname{prob}(x_i)
  = \frac{G(x_i)}{\sum_{j=1}^{\text{pop\_size}} G(x_j)}.
  $$

- 适应度越高的个体，在交配池中出现的次数越多。

示例：初始种群（第 $0$ 代）及其适应度和被选概率：

| $i$ | 染色体 Chromosome $x^i$ | $G(x^i)$ | $\operatorname{Prob}(x^i)$ | 说明 |
|-----|-------------------------|---------|----------------------------|------|
| 1   | 1011                    | 56      | 0.55                       | 最大概率 Maximum probability |
| 2   | 0010                    | 29      | 0.28                       |      |
| 3   | 0001                    | 16      | 0.16                       |      |
| 4   | 0000                    | 1       | 0.01                       | 最小概率 Minimum probability |


#### 轮盘赌选择 Roulette Wheel selection

**轮盘赌 Roulette wheel** 常用于实现比例选择：

- 每个个体在“轮盘”上占据的区间与其适应度成正比。  
- 转动轮盘，相当于按概率随机选择个体。

具体步骤：

1. 计算种群中所有个体适应度之和：

   $$
   \text{Sum} = \sum_{i=1}^{\text{pop\_size}} G(x_i) = 102.
   $$

2. 计算累积适应度：

   - $S_1 = G_1$  
   - $S_2 = G_1 + G_2$  
   - $S_3 = G_1 + G_2 + G_3$  
   - $S_4 = G_1 + G_2 + G_3 + G_4 = \text{Sum}$  

   示例：  
   $0,\ 56\ (S_1),\ 85\ (S_2),\ 101\ (S_3),\ 102\ (S_4=\text{Sum})$

3. 在 $[0,\ \text{Sum})$ 上产生一个随机数 $rand$。  
   根据 $rand$ 所在区间选择个体：

   - 若 $rand < S_1$，选个体 1；  
   - 若 $S_1 \le rand < S_2$，选个体 2；  
   - 若 $S_2 \le rand < S_3$，选个体 3；  
   - 若 $S_3 \le rand < \text{Sum}$，选个体 4。

多次重复上述过程，就能根据各自概率构造出**交配池 Mating pool**。


#### 从种群到交配池 From population to mating pool

在比例选择下，从种群到交配池的一个示例：

| $i$ | $x^i$ | $G(x^i)$ | $\operatorname{Prob}(x^i)$ | Mating pool | $G(x^i)$ |
|-----|-------|----------|----------------------------|-------------|----------|
| 1   | 1011  | 56       | 0.55                       | 1011        | 56       |
| 2   | 0010  | 29       | 0.28                       | 0010        | 29       |
| 3   | 0001  | 16       | 0.16                       | 1011        | 56       |
| 4   | 0000  | 1        | 0.01                       | 0001        | 16       |

可以看到适应度更高的染色体 1011 在交配池中出现两次。

#### 选择的作用 The effect of selection

- 总体趋势：**更好的个体 Better individuals**在交配池中出现的拷贝数更多。

从两个角度看作用：

1. **最优性 Optimality**

   - 若交配池中的“平均质量”高于原始种群：  
     **Mating pool > Previous population**  
   - 选择将搜索引导到搜索空间中更有希望的区域：  
     **Selection guides the search into a promising area in the search space**。

2. **多样性 Diversity**

   - 交配池中的候选解种类少于原始种群：  
     **Mating pool < Previous population**  
   - 选择会减少候选解的多样性：  
     **Selection decreases the number of different candidates**。


#### 仅有选择的局限性 Limitation of selection alone

若只有“选择”这个操作，在多代之后：

- 种群中的所有成员最终都会变成同一个个体——  
  初始种群中最好的那一个。  

因此：

- **仅靠选择 Selection alone 不能解决优化问题 cannot solve the optimization problem**。  
- 还必须结合 **交叉 Crossover** 与 **变异 Mutation** 等算子，才能持续产生新解并跳出局部最优。

---
## GA 交叉与变异 Crossover & Mutation

### 交叉 Crossover

**新解 New solutions** 通过 **交叉 Crossover** 和 **变异 Mutation** 操作产生。

这些算子具有以下特点：

- 直接作用在**个体 Individuals**上。  
- 通常在**选择 Selection** 之后使用。  
- 在大多数情况下，与**适应度 Fitness**没有直接关系（只依赖于编码本身）。

### 交叉操作的一般形式 General crossover operation

在交叉操作中，两个个体（父代）以某种方式互相交换染色体片段：

- **简单单点交叉 Simple (One-Point) Crossover**
- **多点交叉 K-Point Crossover**

示意（单点交叉前后）：

- 交叉前 Before crossover  
  - Parent 1：$1\ 0\ \color{red}{0\ 0}$  
  - Parent 2：$0\ 1\ \color{red}{0\ 1}$
- 交叉后 After crossover  
  - Child 1：$1\ 0\ \color{red}{0\ 1}$  
  - Child 2：$0\ 1\ \color{red}{0\ 0}$  

![](Assets/Pasted%20image%2020251212232737.png)

### 简单单点交叉 Simple (One-Point) Crossover

- 从交配池 **Mating pool** 中随机选择两个个体作为父代。  
- 随机选择一个**交叉位置 Crossover site**。  
- 在该位置之后的基因片段互换，产生两个子代。

示例（交叉位置为第 $4$ 位）：

- Parent 1：$1\ 0\ 0\ 1\ |\ 1\ 0$  
- Parent 2：$0\ 0\ 1\ 0\ |\ 0\ 1$  

交叉后：

- Child 1：$1\ 0\ 0\ 1\ |\ 0\ 1$  
- Child 2：$0\ 0\ 1\ 0\ |\ 1\ 0$  


### 多点交叉 K-Point Crossover

**K 点交叉 K-point crossover**：选择 $k$ 个交叉位置，将这些位置之间的子串成段交换。

- 交叉点越多，重组方式越复杂。  
- 示例（$K = 2$，交叉点在第 $2$ 与第 $4$ 位）：

  - Parent 1：$1\ 0\ |\ 0\ 1\ |\ 1\ 0$  
  - Parent 2：$0\ 0\ |\ 1\ 0\ |\ 0\ 1$

  交换中间子串后：

  - Child 1：$1\ 0\ |\ 1\ 0\ |\ 1\ 0$  
  - Child 2：$0\ 0\ |\ 0\ 1\ |\ 0\ 1$


### 交叉概率 Crossover rate

设交叉概率为 $P_c$：

1. 从交配池中成对选取父代。  
2. 生成一个 $[0,1]$ 区间内的随机数 $R_c$。  
3. 若 $R_c \le P_c$，则这对父代执行交叉；  
   否则不执行交叉，父代直接作为后代。  

典型取值：$P_c \in [0.6, 0.9]$。

- 当 $R_c \le P_c$：会生成新的重组染色体，适应度 $G(x)$ 可能变大也可能变小。  
- 当 $R_c > P_c$：该对个体保持不变，直接进入下一代。


### 交叉的影响 Effects of crossover

- 在有限样本的情况下，交叉可能**提高或降低平均适应度 Average fitness**。  
- 更重要的是：交叉显著增加种群的**多样性 Diversity**，通过重组基因探索新的搜索区域。

例子（部分数据）：

| Mating Pool | $G(x^i)$ | 交叉后 $x^i_{\text{cross}}$ | $G(x^i_{\text{cross}})$ |
|------------|---------|-----------------------------|-------------------------|
| 1011       | 56      | 1010                        | 61  |
| 0010       | 29      | 0011                        | 40  |
| 1011       | 56      | 1011                        | 56  |
| 0001       | 16      | 0001                        | 16  |

可以看到部分个体适应度提高，部分保持不变。

### 交叉与早熟收敛 Crossover & Premature convergence

在某些情况下，如果种群已经高度相似，交叉只会在非常相近的染色体之间交换片段：

- 交叉前后产生的子代与父代几乎相同，无法产生真正新的个体。  
- 此时，仅靠**选择 Selection** + **交叉 Crossover** 仍然无法解决优化问题，会导致**早熟收敛 Premature convergence**。

因此，需要引入**变异 Mutation**算子。


## 变异 Mutation

在交叉之后，对个体中的基因施加**随机扰动 Random perturbation**：

- 在每个基因上，以一个较小的**变异概率 Mutation probability** $P_m$ 发生变异。  
  - 常见范围：$P_m \approx 0.01 \sim 0.001$。  

- 变异策略很多，这里以**二进制串 Binary string**为例：  
  - 对于每个位，按概率 $P_m$ 将 $0$ 变成 $1$，或将 $1$ 变成 $0$。  

变异的作用：

- 保证不断**产生新个体 Generate new individuals**（几乎必然）。  
- 帮助算法跳出局部最优，缓解早熟收敛问题。  
- 与交叉一起，共同维持种群的多样性并探索搜索空间。


---
## GA 停止准则与总结 Stopping Criterion & Summary

### 停止条件 Stopping condition

- 如果**停止条件 Stopping condition**没有满足，遗传算法 **GA** 会继续产生新的世代 New generations。

常见的停止条件 Some stopping conditions：

- **最大迭代代数 Maximum number of generations**  
- **得到满意的解 Satisfactory solution obtained**  
- **种群收敛 Population converges**

### 进化机制 Evolution mechanism

#### 交叉与变异 Crossover and mutation

- 提升种群多样性 **Increase the population diversity**  
- 与适应度无直接关系 **Has nothing to do with the fitness**  
- 侧重于产生新个体 **Focus on generating new individuals**

#### 选择 Selection

- 降低种群多样性 **Decrease the population diversity**  
- 提高适应度 **Increase the fitness**  
- 引导进入有前景的区域 **Focus on entering promising areas**


**Advantages of GAs**

- 是一种通用的优化方法 **General optimization method**，与具体**问题领域 Problem domain**无关。  
- 使用**种群 Population**进行搜索，可以并行评估候选解 **Candidate solutions**。  
- 具有产生多种最优/近似最优解的能力 **Ability to generate various kinds of optimal solutions**。  
- 算法流程相对容易实现 **The process is not difficult to implement**。


**Disadvantages of GAs**

- 可能无法获得“非常高质量”的最优解 **May not obtain a very high-quality optimal solution**。  
- 算法包含多个**参数 Parameters**，这些参数的选择会显著影响解的质量 **Quality of the solution**。  
- 计算速度不快 **It is not fast**，在大规模问题上可能较耗时。
