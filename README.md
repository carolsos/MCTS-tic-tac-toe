# MCTS-tic-tac-toe
## MCTS是什么？

MCTS，即蒙特卡洛树搜索（Monte Carlo Tree Search），是一种用于某些决策过程的启发式搜索算法，特别适用于有明确结束条件和有限可能动作的领域，如棋类游戏。它通过构建一棵树，每个节点代表一个游戏状态，边代表一个可能的动作，来模拟可能的游戏进程并评估结果。

MCTS的基本步骤包括：

选择（Selection）：从根节点开始，根据一定的策略（如UCB公式）选择最有前景的子节点，直到达到叶子节点。

扩展（Expansion）：在叶子节点处添加一个或多个子节点，代表可能的下一步动作。

模拟（Simulation）：从新添加的子节点开始，进行随机模拟（也称为“playout”或“rollout”），直到游戏结束。

反向传播（Backpropagation）：将模拟的结果更新到路径上的所有节点。


