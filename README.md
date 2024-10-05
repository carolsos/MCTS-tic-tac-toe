# MCTS-tic-tac-toe
## MCTS是什么？

MCTS，即蒙特卡洛树搜索（Monte Carlo Tree Search），是一种用于某些决策过程的启发式搜索算法，特别适用于有明确结束条件和有限可能动作的领域，如棋类游戏。它通过构建一棵树，每个节点代表一个游戏状态，边代表一个可能的动作，来模拟可能的游戏进程并评估结果。

MCTS的基本步骤包括：

选择（Selection）：从根节点开始，根据一定的策略（如UCB公式）选择最有前景的子节点，直到达到叶子节点。

扩展（Expansion）：在叶子节点处添加一个或多个子节点，代表可能的下一步动作。

模拟（Simulation）：从新添加的子节点开始，进行随机模拟（也称为“playout”或“rollout”），直到游戏结束。

反向传播（Backpropagation）：将模拟的结果更新到路径上的所有节点。

```python

import random
import numpy as np


class GameState:
    def __init__(self, player='X', opponent='O'):
        self.player = player
        self.opponent = opponent
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.available_moves = []
        self.winner = None

        # Initialize available moves
        for i in range(3):
            for j in range(3):
                self.available_moves.append((i, j))

    def is_valid_move(self, row, col):
        return self.board[row][col] == ' '

    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.board[row][col] = self.player
            # Update available moves
            self.available_moves = []
            for i in range(3):
                for j in range(3):
                    if self.board[i][j] == ' ':
                        self.available_moves.append((i, j))
            if self.check_winner(row, col):
                self.winner = self.player
            else:
                self.switch_player()

    def switch_player(self):
        self.player, self.opponent = self.opponent, self.player

    def check_winner(self, row, col):
        # Check rows, columns, and diagonals
        win_conditions = [
            [self.board[0][0], self.board[0][1], self.board[0][2]],
            [self.board[1][0], self.board[1][1], self.board[1][2]],
            [self.board[2][0], self.board[2][1], self.board[2][2]],
            [self.board[0][0], self.board[1][0], self.board[2][0]],
            [self.board[0][1], self.board[1][1], self.board[2][1]],
            [self.board[0][2], self.board[1][2], self.board[2][2]],
            [self.board[0][0], self.board[1][1], self.board[2][2]],
            [self.board[0][2], self.board[1][1], self.board[2][0]],
        ]
        for condition in win_conditions:
            if condition[0] == condition[1] == condition[2]!= ' ':
                return True
        return False

    def is_draw(self):
        return len(self.available_moves) == 0 and not self.winner

    def clone(self):
        new_state = GameState(self.player, self.opponent)
        new_state.board = [row[:] for row in self.board]
        new_state.available_moves = self.available_moves[:]
        new_state.winner = self.winner
        return new_state

    def __str__(self):
        board_str = '\n'.join([' '.join(row) for row in self.board])
        available_moves_str = ', '.join([f'({r},{c})' for r, c in self.available_moves])
        return f"Board:\n{board_str}\nAvailable Moves: {available_moves_str}\nWinner: {self.winner}"


class Node:
    def __init__(self, game_state, parent=None):
        self.game_state = game_state
        self.parent = parent
        self.children = []
        self.wins = 0
        self.visits = 0

    def uct_score(self, exploration_constant=1.4):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration_constant * np.sqrt((2 * np.log(self.parent.visits) / self.visits))

    def best_child(self, exploration_constant=1.4):
        return max(self.children, key=lambda c: c.uct_score(exploration_constant))

    def __str__(self):
        return f"Node with state:\n{self.game_state}\nWins: {self.wins}\nVisits: {self.visits}"


def mcts(root_state, iterations=2000):
    root_node = Node(root_state)

    intermediate_log_file = open('mcts_intermediate_log1.txt', 'w')
    winning_log_file = open('mcts_winning_log.txt', 'w')

    total_simulations = 0
    winning_simulations = 0
    x_wins = 0
    o_wins = 0

    for _ in range(iterations):
        node = root_node
        game_state = node.game_state.clone()

        # Selection
        intermediate_log_file.write(f"Starting selection phase with state:\n{game_state}\n")
        print(f"Starting selection phase with state:")
        print(game_state)

        while node.children and not game_state.is_draw() and not game_state.winner:
            node = node.best_child()
            intermediate_log_file.write(f"Selected child node with state:\n{node}\n")
            print(f"Selected child node with state:")
            print(node)

        # Expansion
        if not game_state.available_moves:
            intermediate_log_file.write("No moves available, returning current best node.\n")
            print("No moves available, returning current best node.")
            return node  # No moves available, game is over or draw

        if not node.children:
            intermediate_log_file.write(f"Expanding node with state:\n{game_state}\n")
            print(f"Expanding node with state:")
            print(game_state)
            move = random.choice(game_state.available_moves)
            new_state = game_state.clone()
            new_state.make_move(*move)
            new_node = Node(new_state, parent=node)
            node.children.append(new_node)
            intermediate_log_file.write(f"Made move {move}, new state:\n{new_node}\n")
            print(f"Made move {move}, new state:")
            print(new_node)

        # Simulation
        current_state = new_state.clone()
        intermediate_log_file.write(f"Starting simulation from state:\n{current_state}\n")
        print(f"Starting simulation from state:")
        print(current_state)
        while not current_state.is_draw() and not current_state.winner:
            move = random.choice(current_state.available_moves)
            current_state.make_move(*move)
            intermediate_log_file.write(f"Simulated move {move}, new state:\n{current_state}\n")
            print(f"Simulated move {move}, new state:")
            print(current_state)
        total_simulations += 1
        if current_state.winner:
            winning_simulations += 1
            if current_state.winner == 'X':
                x_wins += 1
            else:
                o_wins += 1

        # Backpropagation
        while node:
            intermediate_log_file.write(f"Backpropagating from state:\n{game_state}\n")
            print(f"Backpropagating from state:")
            print(game_state)
            if current_state.winner == node.game_state.player:
                node.wins += 1
            node.visits += 1
            node = node.parent

    best_child_node = root_node.best_child()
    if best_child_node:
        best_move = best_child_node.game_state.available_moves[0]
        intermediate_log_file.write(f"Best move found: {best_move}, from state:\n{best_child_node.game_state}\n")
        print(f"Best move found: {best_move}, from state:")
        print(best_child_node.game_state)
        if best_child_node.game_state.winner:
            winning_log_file.write(f"Winning state:\n{best_child_node.game_state}\n")
            print(f"Winning state:\n{best_child_node.game_state}")
        intermediate_log_file.close()
        winning_log_file.close()
        return best_move, total_simulations, winning_simulations, x_wins, o_wins
    else:
        intermediate_log_file.write("No best move found or game is over.\n")
        print("No best move found or game is over.")
        intermediate_log_file.close()
        winning_log_file.close()
        return None, total_simulations, winning_simulations, x_wins, o_wins


# 运行 MCTS
initial_state = GameState()
best_move, total_sim, winning_sim, x_win_count, o_win_count = mcts(initial_state)

# 打印结果
if best_move:
    print("Best move is:", best_move)
else:
    print("No best move found or game is over.")
print(f"Total simulations: {total_sim}")
print(f"Winning simulations: {winning_sim}")
print(f"X wins: {x_win_count}")
print(f"O wins: {o_win_count}")

```

