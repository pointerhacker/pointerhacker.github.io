---
layout: post
title: 一文搞懂FlashAttention
categories: [LLM]
tags: LLM
---

## 简介
FlashAttention的关键创新是借鉴了在线 Softmax的思想，对自注意力计算进行分块处理，从而能够在不访问 GPU 全局内存以获取中间 logits 和注意力分数的情况下，融合整个多头注意力层。在这篇笔记中，我将简要解释为什么对自注意力计算进行分块处理并非易事，以及如何从在线 Softmax 技巧推导出 FlashAttention 的计算过程。我们感谢 Andrew Gu 对这篇笔记进行了校对。

## 1、The Self-Attention

自注意力的计算可以总结为（我们忽略头和批次，因为这些维度上的计算是完全并行的，我们还省略了诸如注意力掩码和缩放因子$$\frac{1}{\sqrt{D}}$$等细节，以简化问题）：

$$\mathbf{O}=\text{softmax}(\mathbf{Q}\mathbf{K}^T)\mathbf{V}$$（1）

其中，$$\mathbf{Q}$$、$$\mathbf{K}$$、$$\mathbf{V}$$和$$\mathbf{O}$$都是形状为$$(L,D)$$的二维矩阵，其中$$L$$是序列长度，$$D$$是每个头的维度（也称为头维度），softmax 应用于最后一个维度（列）。计算自注意力的标准方法是将计算分解为几个阶段：

$$\mathbf{X}=\mathbf{Q}\mathbf{K}^T$$ （2）

$$\mathbf{A}=\text{softmax}(\mathbf{X})$$ （3）

$$\mathbf{O}=\mathbf{A}\mathbf{V}$$ （4）

我们称矩阵$$\mathbf{X}$$为 softmax 之前的 logits，矩阵$$\mathbf{A}$$为注意力分数，矩阵$$\mathbf{O}$$为输出。

FlashAttention 的一个惊人之处在于，我们不需要在全局内存中实现矩阵$$\mathbf{X}$$和$$\mathbf{A}$$，而是将公式 1 中的整个计算过程融合到一个 CUDA 内核中。这要求我们设计一种算法，仔细管理片上内存（类似于流算法），因为 NVIDIA GPU 的共享内存较小。

对于经典的矩阵乘法算法，分块是为了确保片上内存不超过硬件限制。图 1 提供了一个示例。在内核执行期间，无论矩阵的形状如何，片上内存中只存储了$$3T^2$$个元素。这种分块方法是有效的，因为加法是结合的，允许将整个矩阵乘法分解为许多分块矩阵乘法的总和。

然而，自注意力包含一个非结合的 softmax 运算符，这使得像图 1 那样简单地对自注意力进行分块变得困难。有没有办法使 softmax 具有结合性呢？

![image-20250224163254633](http://pointerhacker.github.io/imgs/posts/flash0.02/image-20250224163254633.png)

图 1.该图简要解释了如何对矩阵乘法$$\mathbf{C}=\mathbf{A}\mathbf{B}$$的输入和输出矩阵进行分块，矩阵被划分为$$T\times T$$的分块。对于每个输出分块，我们从左到右扫描矩阵$$\mathbf{A}$$中相关的分块从，上到下扫描矩阵$$\mathbf{B}$$中相关的分块，并将值从全局内存加载到片上内存（以蓝色标记，总的片上内存占用为$$O(T^2)$$）。对于分块的部分矩阵乘法，在位置$$(i,j)$$，我们从片上内存中加载分块内的所有$$k$$的$$\mathbf{A}[i,k]$$和$$\mathbf{B}[k,j]$$（以红色标记），然后在片上内存中将$$\mathbf{A}[i,k]\times\mathbf{B}[k,j]$$聚合到$$\mathbf{C}[i,j]$$。完成一个分块的计算后，我们将片上内存中的$$\mathbf{C}$$分块写回到主内存，并继续处理下一个分块。实际应用中的分块要复杂得多，你可以参考 Cutlass 在 A100 上的矩阵乘法实现[2]。

## 2 、(Safe) Softmax

让我们先回顾一下 softmax 运算符，以下是 softmax 计算的通用公式：
$$\text{softmax}(\{x_1,\dots,x_N\})=\left(\frac{e^{x_i}}{\sum{j=1}^{N}e^{x_j}}\right)_{i=1}^{N}\quad(5)$$
注意，$$x_i$$可能非常大，而$$e^{x_i}$$很容易溢出。

例如，float16 能支持的最大数字是 65536，这意味着对于$$x>11$$，$$e^x$$会超出 float16 的有效范围。为了缓解这一问题，数学软件通常采用一种被称为 safe softmax 的技巧：
$$\frac{e^{x_i}}{\sum_{j=1}^{N}e^{x_j}}=\frac{e^{x_i-m}}{\sum_{j=1}^{N}e^{x_j-m}}\quad(6)$$
其中$$m=\max_{j=1}^{N}(x_j)$$，这样我们可以确保每个$$x_i-m\leq 0$$，这是安全的，因为指数运算符对于负输入是准确的。

然后我们可以将安全 softmax 的计算总结为以下 3 步算法：



### Algorithm 3-pass safe softmax

算法：3-pass safe softmax

符号：

• $$\{m_i\}$$: $$\max_{j=1}^{i}\{x_j\}$$，初始值$$m_0=-\infty$$。

• $$\{d_i\}$$: $$\sum_{j=1}^{i}e^{x_j-m_N}$$，初始值$$d_0=0$$，$$d_N$$是  safe-softmax 的分母。

• $$\{a_i\}$$: 最终的 softmax 值。

主体：
$$
\begin{aligned}
&\text{for } i = 1 \text{ to } N \text{ do} \\
&\hspace{1cm} m_i = \max(m_{i-1}, x_i) \quad \text{(7)} \\
&\text{end} \\
&\text{for } i = 1 \text{ to } N \text{ do} \\
&\hspace{1cm} d_i = d_{i-1} + e^{x_i - m_N} \quad \text{(8)} \\
&\text{end} \\
&\text{for } i = 1 \text{ to } N \text{ do} \\
&\hspace{1cm} a_i = \frac{e^{x_i - m_N}}{d_N} \quad \text{(9)} \\
&\text{end}
\end{aligned}
$$



该算法需要我们对$$[1,N]$$进行三次迭代。在 Transformer 中的自注意力上下文中，$$\{x_i\}$$是由$$\mathbf{Q}\mathbf{K}^T$$计算出的预 softmax logit。这意味着如果我们没有所有 logit$$\{x_i\}_{i=1}^{N}$$（我们的 SRAM 不够大，无法容纳它们），我们需要访问$$\mathbf{Q}$$和$$\mathbf{K}$$三次（实时重新计算 logit），这在 I/O 上是低效的。



## 3 Online Softmax

如果我们把方程7、8和9融合到一个循环中，我们可以将全局内存访问时间从3减少到1。不幸的是，我们不能将方程7和8融合到同一个循环中，因为8依赖于$$m_N$$，而mN直到第一个循环完成才能确定。

​	我们可以创建另一个序列$$d_i^‘:=\sum_{j=1}^{i}e^{x_j-mi}$$作为原始序列$$d_i:=\sum_{j=1}^{i}e^{x_j-m_N}$$的替代，以消除对N的依赖，这两个序列的第N项是相同的：$$d_N=d_N^‘$$，因此我们可以安全地将方程9中的$$d_N$$替换为$$d_N^‘$$。我们还可以找到$$d_i^‘$$和$$d_{i-1}^‘$$之间的递推关系：



$$\begin{aligned} d'_i = \sum_{j=1}^i e^{x_j-m_i} \\ = \left(\sum_{j=1}^{i-1} e^{x_j-m_i}\right) + e^{x_i-m_i} \\= \left(\sum_{j=1}^{i-1} e^{x_j-m_{i-1}}\right)e^{m_{i-1}-m_i} + e^{x_i-m_i} \\ = d'_{i-1}e^{m_{i-1}-m_i} + e^{x_i-m_i} \end{aligned}$$



### Algorithm 2-pass online softmax

for $i \leftarrow 1,N$ do

​	 $$m_i \leftarrow \max(m_{i-1}, x_i)$$ 

​	$$d_i' \leftarrow d_{i-1}' e^{m_{i-1}-m_i} + e^{x_i-m_i}$$ 

end

for $i \leftarrow 1,N$ do

​	 $$a_i \leftarrow \frac{e^{x_i-m_N}}{d_N'}$$ 

end



## 4 FlashAttention

不幸的是，对于softmax，答案是否定的，但在自注意力机制中，我们的最终目标不是注意力分数矩阵A，而是O矩阵，它等于A乘以V。我们能否为O找到一种单次遍历的递推形式呢？

让我们尝试将自注意力计算的第k行（所有行的计算都是独立的，为了简单起见，我们解释一行的计算）表述为递推算法：

Algorithm Multi-pass Self-Attention

- $$Q[k,:]$$：表示 $$Q$$ 矩阵的第 $$k$$ 行向量
- $$K^T[:,i]$$：表示 $$K^T$$ 矩阵的第 $$i$$ 列向量
- $$O[k,:]$$：表示输出 $$O$$ 矩阵的第 $$k$$ 行
- $$V[i,:]$$：表示 $$V$$ 矩阵的第 $$i$$ 行
- $$\{o_i\}: \sum_{j = 1} a_jV[j,:]$$，一个存储部分聚合结果 $$A[k,:i] \times V[i,:]$$ 的行向量

for $i \leftarrow 1,N$ do
$$
\begin{align*}
x_i &\leftarrow Q[k,:]K^T[:,i]\\
m_i &\leftarrow \max(m_{i - 1},x_i)\\
d_i' &\leftarrow d_{i - 1}'e^{m_{i - 1}-m_i}+e^{x_i - m_i}
\end{align*}
$$

End

for $i \leftarrow 1,N$ do
$$
\begin{align}
a_i &\leftarrow \frac{e^{x_i - m_N}}{d_N'}\tag{11}\\
o_i &\leftarrow o_{i - 1}+a_iV[i,:]\tag{12}
\end{align}
$$

End
$$O[k,:] \leftarrow o_N$$ 



1. 让我们用方程11中的定义替换方程12中的$$a_i$$：
$$
\boldsymbol{o}_i := \sum_{j = 1}^{i} \left( \frac{e^{x_j - m_N}}{d_N'} V[j,:] \right) \tag{13}
$$
2. 这仍然依赖于$$m_N$$和$$d_N$$，而这两个值直到前一个循环完成才能确定。但我们可以在第3节中再次使用“替代”技巧，创建一个替代序列$$o'$$：
$$
\boldsymbol{o}_i' := \left( \sum_{j = 1}^{i} \frac{e^{x_j - m_i}}{d_i'} V[j,:] \right)
$$
3. 序列$o_i$和$$o'$$的第n个元素是相同的：$$o_N^‘=o_N$$，并且我们可以找到$$o_i'$$和$$o_{i-1}'$$之间的递推关系：：

$$
\begin{align*}
\boldsymbol{o}_i' &= \sum_{j = 1}^{i} \frac{e^{x_j - m_i}}{d_i'} V[j,:] \\
&= \left( \sum_{j = 1}^{i - 1} \frac{e^{x_j - m_i}}{d_i'} V[j,:] \right) + \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \left( \sum_{j = 1}^{i - 1} \frac{e^{x_j - m_{i - 1}}}{d_{i - 1}'} \frac{e^{x_j - m_i}}{e^{x_j - m_{i - 1}}} \frac{d_{i - 1}'}{d_i'} V[j,:] \right) + \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \left( \sum_{j = 1}^{i - 1} \frac{e^{x_j - m_{i - 1}}}{d_{i - 1}'} V[j,:] \right) \frac{d_{i - 1}'}{d_i'} e^{m_{i - 1} - m_i} + \frac{e^{x_i - m_i}}{d_i'} V[i,:] \\
&= \boldsymbol{o}_{i - 1}' \frac{d_{i - 1}' e^{m_{i - 1} - m_i}}{d_i'} + \frac{e^{x_i - m_i}}{d_i'} V[i,:] \tag{14}
\end{align*}
$$



这仅依赖于$$d_i^{'}$$,$$d_{i-1}^{'}$$,$$m_i$$,$$m_{i-1}$$和$$x_i$$，因此我们可以将自注意力中的所有计算融合到一个循环中：

for $i \leftarrow 1,N$ do

   - $$x_{i} \leftarrow Q[k,:]K^{T}[:,i]$$
   - $$m_{i} \leftarrow \max(m_{i - 1},x_{i})$$
   - $$d_{i}' \leftarrow d_{i - 1}'e^{m_{i - 1}-m_{i}}+e^{x_{i}-m_{i}}$$
   - $$o_{i}' \leftarrow o_{i - 1}'\frac{d_{i - 1}'e^{m_{i - 1}-m_{i}}}{d_{i}'}+\frac{e^{x_{i}-m_{i}}}{d_{i}'}V[i,:]$$

end
- $$O[k,:]\leftarrow o_{N}'$$ 

状态$$x_i$$,$$m_i$$,$$d_i^{'}$$,和$$o_i^{'}$$的占用空间较小，可以轻松地放入 GPU 共享内存中。由于该算法中的所有操作都是可结合的，因此它与分块兼容。如果我们按分块计算状态，该算法可以表示如下：

算法 FlashAttention（分块）

新符号说明


• $$b$$：分块的块大小

• $$\#tiles$$：行中的分块数量，$$N=b\times\#tiles$$

• $$x_i$$：一个向量，存储第$$i$$个分块的$$Q[k]K^T$$值，范围为$$[(i-1)b:i b]$$

• $$m_i^{(local)}$$：$$x_i$$内部的局部最大值



for $i \leftarrow 1,N$ do

- $$\boldsymbol{x}_{i} \leftarrow Q[k,:]K^{T}[:, (i - 1)b:ib]$$
- $$m_{i}^{(\text{local})}=\max_{j = 1}^{b}(\boldsymbol{x}_{i}[j])$$
- $$m_{i} \leftarrow \max\left(m_{i - 1},m_{i}^{(\text{local})}\right)$$
- $$d_{i}' \leftarrow d_{i - 1}'e^{m_{i - 1}-m_{i}}+\sum_{j = 1}^{b}e^{\boldsymbol{x}_{i}[j]-m_{i}}$$
- $$\boldsymbol{o}_{i}' \leftarrow \boldsymbol{o}_{i - 1}'\frac{d_{i - 1}'e^{m_{i - 1}-m_{i}}}{d_{i}'}+\sum_{j = 1}^{b}\frac{e^{\boldsymbol{x}_{i}[j]-m_{i}}}{d_{i}'}V[j+(i - 1)b,:]$$

end
   - $$O[k,:]\leftarrow \boldsymbol{o}_{N/b}'$$ 



图 2 展示了如何将此算法映射到硬件上。

![image-20250224201404115](http://pointerhacker.github.io/imgs/posts/flash0.02/image-20250224201404115.png)

图 2.上图展示了 FlashAttention 在硬件上的计算方式。蓝色块表示存储在 SRAM 中的分块，而红色块对应第$$i$$行。$$L$$表示序列长度，非常可能大（例如 16k），$$D$$表示 Transformer 中的头维度，通常较小（例如 GPT3 中为 128），$$B$$是可以控制的块大小。值得注意的是，SRAM 的总占用空间仅取决于$$B$$和$$D$$，与$$L$$无关。因此，该算法可以扩展到长上下文，而不会遇到内存问题（GPU 共享内存较小，H100 架构为 228kb/SM）。在计算过程中，我们从左到右扫描$$K^T$$和$$A$$的分块，从上到下扫描$$V$$的分块，并相应更新$$m$$、$$d$$和$$O$$的状态。


https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
