---
layout: post
title: gymnasium入门指南
categories: [RL]
tags: RL
---
## 基本概念

### 是什么

Gymnasium 是一个为所有单**代理强化学习环境提供 API**（应用程序编程接口）的项目，并实现了常见环境：cartpole、pendulum、mountain-car、mujoco、atari 等。本页将概述如何使用 Gymnasium 的基础知识，包括其四个关键函数： [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make) 、 [`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset) 、 [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)和[`Env.render()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) 。Gymnasium 的核心是[`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env) ，它是一个高级 Python 类，**代表强化学习理论中的马尔可夫决策过程 (MDP)**（注意：这不是完美的重构，缺少 MDP 的几个组件）。该类为用户提供了**生成初始状态、根据给定操作转换/移动到新状态**以及可视化环境的能力。除了[`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env)之外，还提供了[`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)来帮助增强/修改环境，特别是代理观察、奖励和采取的行动。

> Gymnasium提供ENV将马尔可夫与环境交互过程封装起来通过API的形式提供给用户，
>
> 提供了[`Wrapper`](https://gymnasium.farama.org/api/wrappers/#gymnasium.Wrapper)来帮助增强/修改环境

### 初始化环境

在 Gymnasium 中初始化环境非常简单，可以通过[`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)函数完成：

```python
import gymnasium as gym
env = gym.make('CartPole-v1')
```

该函数将返回一个[`Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env)供用户交互。要查看您可以创建的所有环境，请使用[`pprint_registry()`](https://gymnasium.farama.org/api/registry/#gymnasium.pprint_registry) 。此外， [`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)还提供了许多附加参数，用于指定环境关键字、添加更多或更少的包装器等。有关更多信息，请参阅[`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make) 。

### 与环境互动

在强化学习中，下图所示的经典“代理-环境循环”是主体和环境如何相互作用的简化表示。代理接收对环境的观察，然后选择一个操作，环境使用该操作来确定奖励和下一个观察。然后循环重复，直到环境结束（终止）。

<img src="http://pointerhacker.github.io/imgs/posts/gymnasium/AE_loop.png" alt="../../_images/AE_loop.png" style="zoom:8%;" />

Show me in code 看看智能体和环境的交互过程

```python
import gymnasium as gym
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time

# 创建环境
env = gym.make("LunarLander-v3", render_mode='rgb_array')

# 重置环境
obs = env.reset()

# 渲染初始状态
img = plt.imshow(env.render()) # 使用维持图像对象的方式
plt.axis('off')

done = False

while not done:
    # 采取随机动作
    action = env.action_space.sample()
    
    # 执行动作
    obs, reward, done, truncated, info = env.step(action)
    
    # 渲染并更新显示图像
    img.set_data(env.render())
    plt.axis('off')
    
    # 更新图像
    display(img.figure)
    clear_output(wait=True)
    
    # 生产出视频效果(可选)
    time.sleep(0.05)
    
env.close()
```

输出类似这样

<img src="http://pointerhacker.github.io/imgs/posts/gymnasium/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif" alt="https://user-images.githubusercontent.com/15806078/153222406-af5ce6f0-4696-4a24-a683-46ad4939170c.gif" style="zoom:25%;" />

首先，使用[`make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)创建一个环境，并附加一个关键字`"render_mode"` ，指定如何可视化环境。有关不同渲染模式的默认含义的详细信息，请参阅[`Env.render()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.render) 在此示例中，我们使用`"LunarLander"`环境，其中代理控制需要安全着陆的宇宙飞船。初始化环境后，我们[`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)环境以获得对环境的第一次观察以及附加信息。要使用特定的随机种子或选项初始化环境（请参阅环境文档以了解可能的值），请使用带有`reset()` `seed`或`options`参数。由于我们希望继续代理-环境循环，直到环境结束（时间步数未知），因此我们将`done`定义为一个变量，以了解何时停止与环境交互以及使用它的 while 循环。接下来，代理在环境中执行操作， [`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)执行所选操作（在本例中是随机的`env.action_space.sample()` ）以更新环境。这个动作可以想象为移动机器人或按下游戏控制器上的按钮，从而导致环境发生变化。结果，代理从更新的环境中收到新的观察结果以及采取行动的奖励。例如，这种奖励可能对消灭敌人是积极的，或者对进入熔岩是消极的奖励。一种这样的动作-观察交换被称为**timestep【时间步】**。然而，经过一些时间步长后，环境可能会结束，这称为最终状态。例如，机器人可能已经崩溃，或者可能已经成功完成任务，环境将需要停止，因为代理无法继续。在 Gymnasium 中，如果环境已终止，则由`step()`返回作为第三个变量`terminated` 。同样，我们可能还希望环境在固定数量的时间步后结束，在这种情况下，环境会发出截断的信号。如果`terminated`或`truncated`为`True`那么我们结束这一集，但在大多数情况下，用户可能希望重新启动环境，这可以通过`env.reset()`来完成。

### 行动和观察空间

每个环境都使用[`action_space`](https://gymnasium.farama.org/api/env/#gymnasium.Env.action_space)和[`observation_space`](https://gymnasium.farama.org/api/env/#gymnasium.Env.observation_space)属性指定有效操作和观察的格式。这有助于了解环境的预期输入和输出，因为所有有效的操作和观察都应包含在各自的空间中。在上面的示例中，我们通过`env.action_space.sample()`对随机操作进行采样，而不是使用代理策略，将观察结果映射到用户想要执行的操作。重要的是， `Env.action_space`和`Env.observation_space`是`Space`的实例，这是一个高级 python 类，提供关键函数： `Space.contains()`和`Space.sample()` 。

Gymnasium 支持用户可能需要的各种空间：

- [`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box) ：描述任何n维形状的有上限和下限的有界空间。
- [`Discrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete) ：描述一个离散空间，其中`{0, 1, ..., n-1}`是我们的观察或操作可以采取的可能值。
- [`MultiBinary`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiBinary) ：描述任意n维形状的二元空间。ps:坐标定位
- [`MultiDiscrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.MultiDiscrete) ：由一系列[`Discrete`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Discrete)动作空间组成，每个元素中具有不同数量的动作。
- [`Text`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Text) ：描述具有最小和最大长度的字符串空间。
- [`Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict) ：描述更简单空间的字典。
- [`Tuple`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Tuple) ：描述简单空间的元组。
- [`Graph`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Graph) ：描述具有互连节点和边的数学图（网络）。
- [`Sequence`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Sequence) ：描述更简单的空间元素的可变长度。

### 修改环境

包装器是修改现有环境的便捷方法，而无需直接更改底层代码。使用包装器将使您避免大量重复代码并使您的环境更加模块化。包装器也可以链接起来以组合它们的效果。通过[`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)生成的大多数环境默认情况下已使用[`TimeLimit`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit) 、 [`OrderEnforcing`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.OrderEnforcing)和[`PassiveEnvChecker`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.PassiveEnvChecker)进行包装。为了包装环境，您必须首先初始化基础环境。然后，您可以将此环境与（可能是可选的）参数一起传递给包装器的构造函数：

```python
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation
env = gym.make("CarRacing-v3")
env.observation_space.shape
# out: (96, 96, 3)
wrapped_env = FlattenObservation(env)
wrapped_env.observation_space.shape
# out: (27648,)
```

Gymnasium已经为您提供了许多常用的包装器。
Some examples: 一些例子：

- [`TimeLimit`](https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.TimeLimit) ：如果超过最大时间步数（或者基本环境已发出截断信号），则发出截断信号。
- [`ClipAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.ClipAction) ：剪辑传递到`step`任何操作，使其位于基础环境的操作空间中。
- [`RescaleAction`](https://gymnasium.farama.org/api/wrappers/action_wrappers/#gymnasium.wrappers.RescaleAction) ：对动作应用仿射变换，以线性缩放环境的新下限和上限。
- [`TimeAwareObservation`](https://gymnasium.farama.org/api/wrappers/observation_wrappers/#gymnasium.wrappers.TimeAwareObservation) ：将有关时间步长索引的信息添加到观察中。

如果您有一个包装环境，并且您想要获取所有包装器层下的未包装环境（以便您可以手动调用函数或更改环境的某些底层方面），则可以使用[`unwrapped`](https://gymnasium.farama.org/api/env/#gymnasium.Env.unwrapped)属性。如果环境已经是基础环境， [`unwrapped`](https://gymnasium.farama.org/api/env/#gymnasium.Env.unwrapped)属性将仅返回其自身。

```python
wrapped_env
#<FlattenObservation<TimeLimit<OrderEnforcing<PassiveEnvChecker<CarRacing<CarRacing-v3>>>>>>
wrapped_env.unwrapped
#<gymnasium.envs.box2d.car_racing.CarRacing object at 0x7f04efcb8850>
```



## 训练代理

本页简要概述了如何为 Gymnasium 环境训练代理，特别是，我们将使用 Q-learning 来解决 Blackjack v1 环境。

在我们实现任何代码之前，先概述一下 Blackjack 和 Q-learning。

- Blackjack是最受欢迎的赌场纸牌游戏之一，但也因在某些条件下无法击败而臭名昭著。这个版本的游戏使用无限的牌组（我们抽出有替换的牌），因此在我们的模拟游戏中计算牌不是一个可行的策略。观测结果（observation）是一个元组，这个元组包含了三个元素：其一，玩家当前手牌的点数总和；其二，庄家亮出来的那张牌（正面朝上的牌）的点数；其三，一个布尔值（也就是取值为真或假的逻辑值），用于表明玩家手中是否持有可当作 11 点使用的 A 牌。代理可以在两个操作之间进行选择：停牌 (0)，以便玩家不再拿牌；要牌 (1)，以便玩家拿另一张牌。要获胜，您的牌点数必须大于庄家，但不得超过 21。如果玩家选择停牌或牌点数大于 21，则游戏结束。完整文档可在https://gymnasium.farama.org/environments/toy_text/blackjack找到。
- Q-learning 是 Watkins 于 1989 年提出的一种无模型的离策略学习算法，适用于具有离散动作空间的环境，并因成为第一个证明在某些条件下收敛到最优策略的强化学习算法而闻名。

### 执行一个动作

收到第一个观察结果后，我们将仅使用`env.step(action)`函数与环境交互。该函数将操作作为输入并在环境中执行。因为该操作改变了环境的状态，所以它向我们返回四个有用的变量。例如：

- `next observation` ：这是代理在采取操作后将收到的观察。
- `reward` ：这是代理采取行动后将获得的奖励。
- `terminated` ：这是一个布尔变量，指示环境是否已终止，即由于内部条件而结束。
- `truncated` ：这是一个布尔变量，还指示剧集是否通过提前截断结束，即达到时间限制。
- `info` ：这是一个字典，可能包含有关环境的附加信息。

`next observation` 、 `reward` 、 `terminated`和`truncated`变量是不言自明的，但`info`变量需要一些额外的解释。该变量包含一个字典，其中可能包含一些有关环境的额外信息，但在 Blackjack-v1 环境中您可以忽略它。例如，在 Atari 环境中，信息字典有一个`ale.lives`键，它告诉我们代理还剩下多少条生命。如果特工的生命为 0，那么这一集就结束了。请注意，在训练循环中调用`env.render()`不是一个好主意，因为渲染会大大减慢训练速度。相反，尝试构建一个额外的循环来在训练后评估和展示代理。

### 构建代理

让我们构建一个Q-learning 代理来解决 Blackjack!我们需要一些函数来选择操作并更新代理参数。为了确保代理探索环境，一种可能的解决方案是 epsilon-greedy 策略，其中我们选择一个具有百分比`epsilon`随机操作和贪婪操作（当前被视为最佳） `1 - epsilon` 。

```python
from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Reinforcement Learning agent with an empty dictionary
        of state-action values (q_values), a learning rate and an epsilon.

        Args:
            env: The training environment
            learning_rate: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            discount_factor: The discount factor for computing the Q-value
        """
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure exploration.
        """
        # with probability epsilon return a random action to explore the environment
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # with probability (1 - epsilon) act greedily (exploit)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Updates the Q-value of an action."""
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
```

### 训练代理

为了训练代理，我们将让代理一次玩 one episode 【一集】（一个完整的游戏称为一集），然后在每一集后更新它的 Q 值。智能体必须经历很多episode才能充分探索环境。

```python
# hyperparameters
learning_rate = 0.01
n_episodes = 100_000
start_epsilon = 1.0
epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
final_epsilon = 0.1

env = gym.make("Blackjack-v1", sab=False)
#它的作用是对agent进行封装。
#“累积奖励” 指在某个过程中不断积累的奖励值，
#“情节长度” 可以理解为某个情节、事件或过程的持续时间或长度，
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(
    env=env,
    learning_rate=learning_rate,
    initial_epsilon=start_epsilon,
    epsilon_decay=epsilon_decay,
    final_epsilon=final_epsilon,
)

```

备注：当前的超参数设置为快速训练一个智能体。如果您想收敛到最优策略，请尝试将`n_episodes`增加 10 倍并降低learning_rate（例如降低到 0.001）。

```python
from tqdm import tqdm

for episode in tqdm(range(n_episodes)):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # update the agent
        agent.update(obs, action, reward, terminated, next_obs)

        # update if the environment is done and the current obs
        done = terminated or truncated
        obs = next_obs

    agent.decay_epsilon()
```

可以使用`matplotlib`可视化训练奖励和长度。

```python
from matplotlib import pyplot as plt
# visualize the episode rewards, episode length and training error in one figure
fig, axs = plt.subplots(1, 3, figsize=(20, 8))

# np.convolve will compute the rolling mean for 100 episodes
#Episode Rewards（回合奖励）：
axs[0].plot(np.convolve(env.return_queue, np.ones(100)))
axs[0].set_title("Episode Rewards")
axs[0].set_xlabel("Episode")
axs[0].set_ylabel("Reward")
#Episode Lengths（回合长度）：
axs[1].plot(np.convolve(env.length_queue, np.ones(100)))
axs[1].set_title("Episode Lengths")
axs[1].set_xlabel("Episode")
axs[1].set_ylabel("Length")

#Training Error（训练误差）：
axs[2].plot(np.convolve(agent.training_error, np.ones(100)))
axs[2].set_title("Training Error")
axs[2].set_xlabel("Episode")
axs[2].set_ylabel("Temporal Difference")

plt.tight_layout()
plt.show()
```



### 可视化策略

![img](http://pointerhacker.github.io/imgs/posts/gymnasium/blackjack_with_usable_ace.png)

![img](http://pointerhacker.github.io/imgs/posts/gymnasium/blackjack_without_usable_ace.png)



## 创建自定义环境

本页面提供了如何使用 Gymnasium 创建自定义环境的简短概述，有关渲染的更[完整教程](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/)，请在阅读本页面之前阅读[基本用法](https://gymnasium.farama.org/introduction/basic_usage/)。我们将实现一个非常简单的游戏，称为`GridWorldEnv` ，由固定大小的二维方形网格组成。代理可以在每个时间步长的网格单元之间垂直或水平移动，代理的目标是导航到在剧集开始时随机放置的网格上的目标。

游戏基本信息：

- 观察提供了目标和代理的位置。
- 我们的环境中有 4 个离散动作，分别对应于“右”、“上”、“左”和“下”运动。
- 当代理导航到目标所在的网格单元时，环境结束（终止）。
- 智能体仅在达到目标时才获得奖励，即当智能体达到目标时奖励为 1，否则奖励为零。

### Environment `__init__`

与所有环境一样，我们的自定义环境将继承自[`gymnasium.Env`](https://gymnasium.farama.org/api/env/#gymnasium.Env) ，它定义了环境的结构。环境的要求之一是定义观察和操作空间，它声明环境的可能输入（操作）和输出（观察）的一般集合。正如我们关于游戏的基本信息中所述，我们的代理有四个离散动作，因此我们将使用具有四个选项的`Discrete(4)`空间。对于我们的观察，有几个选项，在本教程中，我们将想象我们的观察看起来像 `{"agent": array([1, 0]), "target": array([0, 3])}` 其中数组元素表示代理或目标的 x 和 y 位置。用于表示观察的替代选项是 2d 网格，其值表示网格上的代理和目标，或 3d 网格，每个“层”仅包含代理或目标信息。因此，我们将观察空间声明为[`Dict`](https://gymnasium.farama.org/api/spaces/composite/#gymnasium.spaces.Dict) ，代理空间和目标空间声明为[`Box`](https://gymnasium.farama.org/api/spaces/fundamental/#gymnasium.spaces.Box) ，允许 int 类型的数组输出。

```python
from typing import Optional
import numpy as np
import gymnasium as gym


class GridWorldEnv(gym.Env):

    def __init__(self, size: int = 5):
        # The size of the square grid
        self.size = size

        # Define the agent and target location; randomly chosen in `reset` and updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = gym.spaces.Discrete(4)
        # Dictionary maps the abstract actions to the directions on the grid
        self._action_to_direction = {
            0: np.array([1, 0]),  # right
            1: np.array([0, 1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, -1]),  # down
        }
```

### 构建观察

由于我们需要在[`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)和[`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)中计算观测值，因此使用`_get_obs`方法将环境状态转换为观测值通常很方便。但是，这不是强制性的，您可以分别计算[`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)和[`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)中的观测值。

```python
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}
```

我们还可以为[`Env.reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)和[`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)返回的辅助信息实现类似的方法。

```python
  def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
```

通常，信息还会包含一些仅在[`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)方法中可用的数据（例如，单独的奖励条款）。在这种情况下，我们必须更新[`Env.step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)中`_get_info`返回的字典。

### Reset函数

由于[`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)的目的是为环境启动一个新的episode，并且有两个参数： `seed`和`options` seed可用于将随机数生成器初始化为确定性状态，options可用于指定重置中使用的值。在重置的第一行，您需要调用`super().reset(seed=seed)` ，它将初始化随机数生成 ( [`np_random`](https://gymnasium.farama.org/api/env/#gymnasium.Env.np_random) ) 以在[`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)的其余部分中使用。在我们的自定义环境中， [`reset()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.reset)需要随机选择代理和目标的位置（如果它们具有相同的位置，我们会重复此操作）。因此，我们可以使用之前实现的`_get_obs`和`_get_info`方法：

```python
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
```

### Step函数

[`step()`](https://gymnasium.farama.org/api/env/#gymnasium.Env.step)方法通常包含环境的大部分逻辑，它接受一个`action`并在应用该操作后计算环境的状态，返回下一个观察的元组，如果环境已终止，则返回结果奖励，如果环境有截断和辅助信息。对于我们的环境，在步骤函数期间需要发生几件事情：

> - 我们使用 self._action_to_direction 将离散动作（eg. 2）转换为具有代理位置移动的网格方向。为了防止代理超出网格范围，我们剪切代理的位置以使其保持在边界内。
> - 我们通过检查代理的当前位置是否等于目标位置来计算代理的奖励。
> - 由于环境不会在内部截断（我们可以在 :meth:make 期间对环境应用时间限制包装器），因此我们将 truncated 永久设置为 False。
> - 我们再次使用_get_obs和_get_info来获取代理的观察和辅助信息。

```python
 def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # An environment is completed if and only if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info
```

### 注册并制作环境

虽然现在可以立即使用新的自定义环境，但更常见的是使用[`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)初始化环境。在本节中，我们将解释如何注册自定义环境然后对其进行初始化。环境 ID 由三个组件组成，其中两个是可选的：可选的命名空间（此处： `gymnasium_env` ）、强制名称（此处： `GridWorld` ）和可选但推荐的版本（此处：v0）。它也可能已注册为`GridWorld-v0` （推荐方法）、 `GridWorld`或`gymnasium_env/GridWorld` ，然后在环境创建期间应使用适当的 ID。入口点可以是字符串或函数，因为本教程不是 python 项目的一部分，我们不能使用字符串，但对于大多数环境，这是指定入口点的正常方法。Register 还有额外的参数，可用于指定环境的关键字参数，例如，是否应用时间限制包装器等。有关更多信息，请参阅[`gymnasium.register()`](https://gymnasium.farama.org/api/registry/#gymnasium.register) 。

```python
gym.register(
    id="gymnasium_env/GridWorld-v0",
    entry_point=GridWorldEnv,
)
```

注册环境后，您可以通过[`gymnasium.pprint_registry()`](https://gymnasium.farama.org/api/registry/#gymnasium.pprint_registry)进行检查，它将输出所有注册的环境，然后可以使用[`gymnasium.make()`](https://gymnasium.farama.org/api/registry/#gymnasium.make)初始化环境。可以使用[`gymnasium.make_vec()`](https://gymnasium.farama.org/api/registry/#gymnasium.make_vec)实例化具有并行运行的同一环境的多个实例的环境的矢量化版本。

```python
import gymnasium as gym
>>> gym.make("gymnasium_env/GridWorld-v0")
<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>
>>> gym.make("gymnasium_env/GridWorld-v0", max_episode_steps=100)
<TimeLimit<OrderEnforcing<PassiveEnvChecker<GridWorld<gymnasium_env/GridWorld-v0>>>>>
>>> env = gym.make("gymnasium_env/GridWorld-v0", size=10)
>>> env.unwrapped.size
10
>>> gym.make_vec("gymnasium_env/GridWorld-v0", num_envs=3)
SyncVectorEnv(gymnasium_env/GridWorld-v0, num_envs=3)
```

### 使用包装器

和其它环境一样

```python
from gymnasium.wrappers import FlattenObservation

>>> env = gym.make('gymnasium_env/GridWorld-v0')
>>> env.observation_space
Dict('agent': Box(0, 4, (2,), int64), 'target': Box(0, 4, (2,), int64))
>>> env.reset()
({'agent': array([4, 1]), 'target': array([2, 4])}, {'distance': 5.0})
>>> wrapped_env = FlattenObservation(env)
>>> wrapped_env.observation_space
Box(0, 4, (4,), int64)
>>> wrapped_env.reset()
(array([3, 0, 2, 1]), {'distance': 2.0})
```



## 参考文档

[Gymnasium Documentation](https://gymnasium.farama.org/)
