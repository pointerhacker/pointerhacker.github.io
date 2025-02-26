---
layout: post
title: 一文搞懂ELO
categories: [LLM]
tags: LLM
---

## 简介
以一种特殊的方式解释[flash attention](https://arxiv.org/abs/2205.14135) ,让你看完后会反思一个问题：

- 我为什么以前没有想到这个？实现起来也“太容易了。

我们首先要了解如何实现标准/普通的attention，然后我们会一一解决效率低下的问题 ----- 好像我们自己要独立地发现 flash attention 一样。

此外，我的子目标是揭开编译器领域社区中的一些术语的神秘面纱：内核、内核融合、实例化等等。

> 注意：我不会解释注意力本身，因为这是指杰伊·阿拉玛（Jay Alammar）的[出色博客](https://jalammar.github.io/illustrated-transformer/)或[我](https://github.com/gordicaleksa/pytorch-original-transformer)。对原transfomer 论文的实现



![image-20250220154300720](http://pointerhacker.github.io/imgs/posts/flashattention/image-20250220154300720.png)

您应该在此博客结束时完美理解此图。

不再废话，让我们从分析论文标题开始：

FlashAttention: **Fast** and **Memory-Efficient** **Exact** Attention with **IO-Awareness**

FlashAttention的核心思想是：

- **Fast**  摘自论文：“我们训练Bert-large（seq-length 512) 比MLPERF 1.1（GPT2）中的训练速度记录快15％（序列长度为 1K）比来自 HuggingFace 和 Megatron-LM 的基线实现快 3 倍。以及 long-range arena（seq.长度 1K-4K）比基线快 2.4 倍。”
- **Memory-efficient** ——与普通注意力机制（其计算量与序列长度呈二次方关系，即O(N²)相比，这种方法的计算量与\(N\)呈亚二次方或线性关系(O(N)。我们稍后会了解其原因和实现方式。
- **Exact**——这意味着它并非是对注意力机制的近似（例如像稀疏方法或低秩矩阵近似方法那样），其输出结果与 “普通” 注意力机制的输出相同。
- **IO aware**——与普通注意力机制相比， flash attention机制是具备输入输出感知能力的。

开玩笑：）- 这只是意味着它不将底层硬件视为黑匣子

相反，它利用了底层硬件（例如GPU，不过其他人工智能加速器也应该可以，这里将以图形处理器作为运行示例）的内存层次结构知识。

让我们进一步扩展 IO awareness部分

虽然更高的FLOPS（计算能力）理论上可以加快程序的运行速度，但在实际运行中，程序的运行时间并不完全取决于计算能力。I/O操作的性能瓶颈可能会限制程序的整体运行时间。

该论文的相关摘录：

虽然这些[approximate]方法将计算需求降低到与序列长度呈线性或接近线性关系，但其中许多方法在标准注意力机制面前并没有显示出实际的加速效果，也没有得到广泛采用。主要原因之一是，它们专注于减少浮点运算次数（这可能与实际加速效果并不相关），并且往往忽视了由内存访问（输入/输出）带来的开销。

什么技巧？

这是硬件：

![image-20250220175343042](http://pointerhacker.github.io/imgs/posts/flashattention/image-20250220175343042.png)

多年来，GPU一直以比增加内存吞吐量（TB/S）更快地增加计算能力 (FLOPS) 

**如果你没有数据要处理，那么即使你能以百亿亿次浮点运算速度进行计算也没有关系。**这两者需要紧密对齐，由于硬件失去了这种平衡，我们必须让我们的软件来进行补偿。

根据计算和内存访问之间的这种比率，操作可以被分类为以下两种：

- **计算受限**（示例：矩阵乘法）
- **内存受限**（例如：逐元素操作[ops]（activation, dropout, masking）、归约操作[ops]（softmax, layer norm, sum, etc）……）

> 关于术语的说明：这个比率通常通过算术强度来衡量，算术强度是指每字节内存访问的算术运算次数。

> 注2：我强烈建议您阅读Horace的博客文章[https://horace.io/brrr_intro.html-](https://horace.io/brrr_intro.html)它将有助于进一步阐明计算/内存/应用限制之间的差异。

事实证明，**attention**（在当前的AI加速器上）**受到内存限制**。

为什么？

因为它“主要由元素ops组成”，或更准确地说，注意力的算术密度不是很高。让我们从论文中放大此图：

![image-20250220180621246](http://pointerhacker.github.io/imgs/posts/flashattention/image-20250217115453044.png)

你可以在左侧栏看到，Mask、softmax 和 dropout 是占用大部分时间的操作，而不是矩阵乘法（尽管大部分浮点运算在矩阵乘法中）。

内存并非一个单一的整体，其本质是分层的，一般规律是：内存速度越快，成本越高，容量越小。

让我们放大图表的这一部分：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*wf7fUgSOUz-9X3-mUWkYmA.png)

> 内存本质上是分层的。
> 我们可以继续扩展这个金字塔，进一步添加固态硬盘（SSD，容量更高，速度更慢）、硬盘驱动器（HDD）等（还有AWS S3？：））。
> 你明白我的意思了。

在实践中，“具备IO意识”归结为利用SRAM比HBM（“高带宽存储器”——名字有点不幸）快得多这一事实，通过减少两者之间的通信来实现。

为了使事情不那么抽象，这里有一个具体的例子：

A100 GPU配备了40-80GB的高带宽存储器（HBM，就是会引发可爱的CUDA内存不足错误的东西），其带宽为1.5-2.0 TB/s，每个108个流式多处理器各有192KB的片上SRAM，带宽估计约为19TB/s。

> 对于H100和其他加速器，类似的比率仍然适用。

现在，我们来看看标准attention实现背后的计算过程：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*9CnVGnPdRrXMQ4VtSmpWtg.png)

你可以看到，标准实现完全无视了硬件的运行方式。它基本上把HBM的加载/存储操作当作零成本（它不是“IO感知”的）。

现在，让我们从第一性原理出发，思考如何使这个实现更高效（时间和内存方面）。

最容易实现的改进是去除多余的HBM读取/写入操作。

为什么要把S写回到HBM，然后再重新加载它来计算softmax呢？我们不妨把它保留在SRAM中，完成所有中间步骤，然后再把最终结果写回到HBM。

这就是编译器领域所说的“内核融合”，它是深度学习中最重要的低级优化之一。

内核本质上是一种“GPU操作”的花哨说法。融合意味着你将多个操作合并在一起。因此，你只从HBM加载一次，执行融合后的操作，然后才将结果写回。通过这样做，你可以减少通信开销。

> 顺便说一句，我真的觉得人们应该停止仅仅因为某个词听起来酷就用它来命名概念。内核是计算机科学领域中被使用得最混乱的词（也许仅次于“模型”）。它可以指任何东西：Linux内核（Linux操作系统的软件核心组件）、神经切线核、SVM核、GPU操作等。它是计算机科学界的希夫·阿拉丁。😂

还有一个你会碰到的术语是“**materialization**”。它指的是在上述标准注意力实现中，我们分配了完整的**NxN矩阵**（S，P）。我们很快会看到，这是Flash Attention直接针对的瓶颈，它将内存复杂度从O(N²)降低到O(N)。

现在背景知识已经完备，让我们深入探究Flash Attention算法。

Flash Attention主要基于两个核心思想：

- 分块（在前向和反向传播中都使用）——基本上是将NxN的softmax/分数矩阵划分为小块。
- 重计算（仅在反向传播中使用——如果你熟悉激活/梯度检查点，这将很容易理解）

仅此而已。

以下是算法的具体内容：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*OJBNIb3fz6JEuisrLfSW6g.png)

我的工作到此结束。希望你喜欢这篇博客文章，未来可以订阅更多内容！🚀

开玩笑的。

让我们再理解一些使分块能够工作的必要概念，然后我会逐行解释这个算法。

# FlashAttention algorithm

使分块方法能够工作的主要障碍是softmax，尤其是softmax将所有分数列耦合在一起的事实。以下是计算softmax的第i个输出的方法。
$$
\sigma(\mathbf{z})i=\frac{e^{z_i}}{\sum_{j=1}^{K}e^{z_j}}
$$
$$z_i$$是第$$i$$个分数（键-查询点积），输出是第$$i$$个标记的概率，我们稍后会用它来加权值向量（同样，我假设你知道注意力机制是如何工作的）。

你看到那个分母了吗？

那就是问题所在。

要计算输入序列中第$$i$$个标记对序列中其他标记的关注程度，你需要将所有这些分数（这里用$$z_j$$表示）都加载到静态随机存取存储器（SRAM）中。

但我得提醒你：SRAM的容量是非常有限的。
你不能直接加载整个内容。
序列长度$$N$$可能是 1000，甚至可以达到 100,000 个标记。因此，$$N^2$$会迅速膨胀。

所以这里有个技巧，我们可以将 softmax 计算分解成更小的块，最终仍然能得到完全相同的结果。

以下是主要公式：

$$m(x) := \max_i x_i, \quad f(x) := [e^{x_1-m(x)} \dots e^{x_B-m(x)}], \quad \ell(x) := \sum_i f(x)_i, \quad \text{softmax}(x) := \frac{f(x)}{\ell(x)}$$

我们可以只提取前B个分数（从$$x_1$$到$$x_B$$），并为它们计算softmax。

这些数字至少目前是不正确的。但请耐心等待，通过迭代，我们将“收敛”到正确的结果。

> 注意：你可以暂时忽略$$m(x)$$部分，至少在我们还处于柏拉图的理念世界中时。它的唯一目的是避免数值不稳定性。在未来的某种更精确的假设性硬件上（例如，我们用更多的比特来表示数据），这将不需要。$$m(x)$$不会以任何方式改变最终结果。
>
> 注意：还可以查看最初引入在线softmax的原始论文：
>
> https://arxiv.org/abs/1805.02867
> https://arxiv.org/abs/2112.05682

现在，关键在于我们可以巧妙地将这些按块计算的部分softmax值结合起来，从而使最终结果实际上是正确的。
主要思路如下：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*ViyWK0-Nc6twYFpxR45OqQ.png)

> 这就是softmax tiling的核心思想。
> 通过在所有块上递归重复这种计算，我们最终得到了正确的softmax输出。

因此，基本上，为了计算属于前2个块（每个块大小为B）的分数的softmax，你需要跟踪每个块的两个统计量：$$m(x)$$（最大分数）和$$l(x)$$（指数分数之和）。

然后，你可以使用归一化系数将它们无缝融合在一起。

> 注意：如果你做一些非常基础的代数运算，你会很容易地相信这些系数是有意义的。
> 通过展开$$f(x)$$和$$l(x)$$项，并乘以$$e^x$$，一些项会简单地相互抵消，这就是基本的操作。

这种逻辑会递归地一直延续到第$$(N/B)$$个块，也就是最后一个块，此时你将得到$$N$$维的正确softmax输出！

好的，我们现在有了理解flash attention算法前向传播所需的所有要素。

> 注意：下面的算法假设我们有一个大小为1的批次（即单个序列）和一个注意力头，我们稍后会很容易地将其扩展（只需在GPU的流式多处理器上进行并行化即可——稍后会详细介绍）。
> 我们暂时忽略dropout和masking，稍后添加会很容易。

![img](http://pointerhacker.github.io/imgs/posts/deepv8/flashv22.png)

让我们逐步分解！

**Step 0:**：HBM（高带宽存储器）的容量是以GB为单位的（例如，RTX 3090有24GB的显存/HBM，A100有40-80GB等），因此分配Q（查询）、K（键）和V（值）并不是问题。

**Step 1:**：让我们计算行/列块大小。
为什么是$$\lceil M/4d\rceil$$？
因为q、k和v 向量都是$$d$$维的，而且我们还需要将它们组合成输出的$$d$$维向量。
所以这个大小基本上允许我们用$$q$$、$$k$$、$$v$$和$$o$$向量最大化 SRAM 容量。

玩具示例：假设$$M=1000$$，$$d=5$$。
在这个例子中，块大小是$$\lceil 1000/(4*5)\rceil=50$$。
所以在这个例子中，我们将一次加载 50 个$$q$$、$$k$$、$$v$$、$$o$$向量的块，以确保我们减少 HBM/SRAM 之间的读写次数。

值得在脑海中保留这张图像（很快就会更有意义）。



![img](http://pointerhacker.github.io/imgs/posts/deepv8/v221*0OV7ituGlfv9EaIfmtqbmg.png)

> As for *B_r*, I’m not exactly sure why do they perform a min op with *d*?
>
> *The author answered: We don't want B_r to be larger than d since we want B_c \* B_r <= M / 4 (we store a few matrices of size B_c \* B_r on SRAM)*

**Step 2:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*zkv3FSgGyZ7-iG_Q4k8GxQ.png)

我们用全零初始化输出矩阵 O。因此，它将作为一个累加器。

同样地，对于 l（记住：它的作用是保存 softmax 的累积分母——即 exp 分数的总和）。m（用于保存行最大分数）被初始化为-inf，因为我们将在其上执行最大值操作，所以无论第一个块的最大值是多少，它肯定比-inf 大，因此这是自然的初始值。

**Step 3:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*VG8bCa1mhiXx6D32wvk7Zw.png)

我们根据第一步中确定的块大小，将 Q、K 和 V 分成块。请参阅上面的图示。

**Step 4:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*wTgKZnd1nFJ1SkRpZ9nj8Q.png)

同样地，将 O、l 和 m 分成块（块大小与 Q 相同）。

**Step 5:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*e0_mx0OSzVNIOE-4ZIq-zw.png)

让我们开始按列循环，即在键/值向量之间循环（即上图中的外层循环）。

**Step 6:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*PaxaJ90G6aHmKP-0RYVrtw.png)

让我们将$$K_j$$和$$V_j$$块从 HBM 加载到 SRAM。记得由于我们构建块大小的方式，在这个时候，SRAM 仍有 50%的空间未被占用（专门用于 Q 和 O）。

![img](http://pointerhacker.github.io/imgs/posts/deepv8/0*Ywl4arvMvN6HCFwZ.png)

> 大致且抽象地说，GPU 的内存布局显然会有所不同。

**Step 7:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*mTCUNX239DWlG-i5LGll9w.png)

开始内层循环，即在查询向量之间循环（同样，参见图示）。

**Step 8:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*y0gyIcenDedgWkmr_s2Bug.png)

将$$Q_i$$($$B_r\times d$$)和$$O_i$$($$B_r\times d$$)块，以及$$l_i$$($$B_r$$)和$$m_i$$($$B_r$$)加载到 SRAM 中。

当我们在计算块大小时，只留有足够的空间用于$$K_j$$、$$V_j$$、$$Q_i$$和$$O_i$$，那么$$l_i$$和$$m_i$$（包括所有中间变量）是如何适应 SRAM 的呢？

我认为答案是：寄存器（可以参考这个 [CUDA 视频](https://www.youtube.com/watch?v=4APkMJdiudU&list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe)系列来了解 GPU 内存层次结构的直观信息）。但我可能错了，如果有在 CUDA 中实现过的人，请纠正我。

🙏我肯定只是通过分析伪算法，遗漏了一些重要的实现细节。

**Step 9:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*LMZEvals-Shf7giT8slQmw.png)

计算$$Q_i$$($$B_r\times d$$)和转置的$$K_j$$($$d\times B_c$$)之间的点积，以获得分数($$B_r\times B_c$$)。正如你所看到的，我们并没有“实现”整个$$N\times N$$的 S（分数）矩阵。只有其中的一部分（$$S{i,j}$$）！

玩具示例：假设外层循环索引是$$j$$（$$j=3$$），内层循环索引是$$i$$（$$i=2$$），$$N$$是 25，块大小是 5，那么我们刚刚计算的就是（假设使用 1 基索引）：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/0*m0pdO7LvXYrZ503F.png)

基本上，这些是我们输入序列中第 6-10 个标记与第 11-15 个标记之间的注意力分数。但重要的是，这些是精确的分数，它们永远不会改变（与会逐渐细化的 softmax 结果不同）。

**Step 10:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*F4LNBA_VsPhqNk4VkmNrOw.png)

使用上一步计算出的分数来计算$$\tilde{m}{i,j}$$、$$\tilde{l}{i,j}$$和$$\tilde{P}{i,j}$$。这很简单。

$$\tilde{m}{i,j}$$是按行计算的，找出上述每一行中的最大元素。

我们通过对元素逐个操作来得到$$\tilde{P}{i,j}$$：

- 归一化——取行最大值并从行分数中减去
- 指数化

 $$\tilde{l}{i,j}$$就是矩阵$$P$$的按行求和。



**Step 11:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*jFlLbR8DS5Ei5NpwgIIvWA.png)

同样很简单，让我们再次使用上面的图示：计算$$m^{new}_i$$和$$l^{new}_i$$。

![img](http://pointerhacker.github.io/imgs/posts/deepv8/0*yHc4h9UJ3AiHQ7fn.png)

$$m_i$$包含之前所有块（$$j=1$$和$$j=2$$，以绿色显示）的行最大值。

$$\tilde{m}_{i,j}$$包含当前块（以黄色显示）的行最大值。

为了得到$$m^{new}_i$$，我们只需要在$$\tilde{m}_{i,j}$$和$$m_i$$之间取最大值。

同样地，计算$$l^{new}_i$$也需要进行乘法操作，正如我们在之前的公式 2 中看到的那样。

**Step 12 (the most important step):**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*KNZwy8pvXo-AwcogXBETng.png)

这是算法中最难的部分，但仍然不算太复杂，特别是当你理解了用于部分 softmax 计算的公式 1 和公式 2 之后。

<img src="http://pointerhacker.github.io/imgs/posts/deepv8/0*2DlPHK6iEltHdQxW.png" alt="img" style="zoom:25%;" />

我们先来分析$$\text{diag}(l)$$这部分。

它本质上只是允许我们以矩阵形式进行按行标量乘法。如果你有一个标量列表$$s$$（大小为$$N$$）和一个矩阵$$A$$（大小为$$N\times N$$），那么$$\text{diag}(s)\times A$$实

际上就是将矩阵$$A$$的每一行与这些标量进行逐元素相乘。

接下来，注意第 12 步与公式 1（为了方便，这里再次列出）之间的相似性。

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*XlUH2kUNmtZzCXMfQ9Q65A.png)

所以，第 12 步的第一项（用绿色下划线标出）的作用是更新当前行中当前块之前的块的 softmax 估计值。如果$$j=1$$（即这是该行的第一个块），第一项将为 0，我们最终只会得到第二项。

第一项乘以$$\text{diag}(l_i)$$是为了抵消前一次迭代中除以同一个常数的操作（这个常数隐藏在$$O_i$$中）。

表达式的第二项（用黄色下划线标出）不需要这种抵消操作，因为正如你所看到的，我们直接将矩阵$$\tilde{P}{i,j}$$与$$V$$向量块（$$V_j$$）相乘。

$$e^x$$项的作用是通过抵消前一次迭代中的$$m$$并用最新的估计值（$$m{\text{new},i}$$）替换它来修改矩阵$$\tilde{P}{i,j}$$和$$O_i$$，这个最新的估计值包含了到目前为止的行最大值。

要说服自己这是有意义的，最简单的方法就是自己模拟几次迭代——如果你仍然不太明白的话。

这真的只需要 5 分钟。这是我的逐步分析（希望它有帮助！）：

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*RuLi2fMrkxeCkPo6_bttvw.png)

正如你所看到的，主要的点是这些在$$P/O$$矩阵外部的$$e$$项会与内部的$$e$$项相互抵消，而我们最终总是得到最新的$$m_{new}^i$$估计值！

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*a4Fu3h_eL9LMLIkLgXW-1g.png)

第三次迭代也是类似的，最终我们得到了正确且最终的结果！

记住：这只是对最终$$O_i$$的当前估计。只有在我们遍历了上图中所有的红色块之后，我们才会得到精确的结果。就是这样！

**Step 13:**

![img](http://pointerhacker.github.io/imgs/posts/deepv8/1*fMJPUiQQ-JhGpZFFOFTNWw.png)

将最新的累积统计值（$$l_i$$和$$m_i$$）写回到 HBM。注意这些的维度是$$B_r$$。

**Steps 14, 15, 16:**

![img](https://miro.medium.com/v2/resize:fit:764/1*GNaqyihDrF2wo3eaFPUCLg.png)

一旦嵌套的 for 循环结束，$$O$$（$$N\times d$$）将包含最终结果：每个输入标记的注意力加权值向量！

就是这样，伙计们。这就是 Flash Attention 的前向传播！

这个算法可以很容易地扩展到“block-sparse  FlashAttention”，这是一种比 FlashAttention 快 2-4 倍的稀疏注意力算法，能够扩展到 64k 的序列长度！其思想是使用块形式的掩码矩阵，我们只需跳过上述嵌套 for 循环中的某些加载/存储操作，通过这种方式，我们可以节省与稀疏系数成比例的计算量。

![img](http://pointerhacker.github.io/imgs/posts/deepv8/0*jBov-kkiUvAv1cA0.png)

假设我们的块长度是 3，在这个示例中，我们只需要进行 3/9 的迭代。因此，速度可以提高 3 倍！
