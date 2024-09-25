import random
import numpy as np

class RL:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = {}
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def save_q_table(self, filename='q_table.txt'):
        with open(filename, 'w') as f:
            for (state, action), value in self.q_table.items():
                f.write(f"{state},{action},{value}\n")  # Save in format: state,action,value

    def load_q_table(self, filename='q_table.txt'):
        try:
            with open(filename, 'r') as f:
                for line in f:
                    state, action, value = line.strip().split(',')
                    # Convert back to tuple for state and action
                    state = eval(state)  # Use eval cautiously; ideally, use ast.literal_eval for safety
                    action = int(action)
                    self.q_table[(state, action)] = float(value)
                    
        except FileNotFoundError:
            print("No previous Q-table found. Starting fresh.")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

    def get_q(self, state, action):
        if (state, action) not in self.q_table:
            self.q_table[(state, action)] = 0  # Initialize Q-value
        return self.q_table[(state, action)]

    def choose_action(self, current_state, available_positions):
        # Explore
        if random.uniform(0, 1) < self.epsilon:
            # Take a random action
            action = np.random.choice(available_positions)
        else:
            # Exploit: choose action with the highest Q-value
            q_values = [self.get_q(current_state, action) for action in available_positions]

            max_q = max(q_values)
            print("max q:", max_q)

            # Choose a random action among the best actions
            best_actions = [action for action in available_positions if self.get_q(current_state, action) == max_q]
            action = np.random.choice(best_actions)

        return action

    def Q_learning(self, state, action, reward, next_state, next_available_actions):
        # Determine future Q-value
        if next_available_actions:
            future_q = [self.get_q(next_state, a) for a in next_available_actions]
            max_future_q = max(future_q)
        else:
            max_future_q = 0  # If there are no future actions, the value is 0

        # Current Q
        current_q = self.get_q(state, action)

        # Update the Q-table
        self.q_table[(state, action)] = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)

        return
