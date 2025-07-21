import numpy as np

# Define the environment (states, actions, rewards)
num_states = 4
num_actions = 2
rewards = np.zeros((num_states, num_actions))
rewards[2, 1] = 1  # Reward of 1 for taking action Right in state 3

# Define the Q-table
Q = np.zeros((num_states, num_actions))

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1 # Exploration rate
num_episodes = 1000

# Q-learning algorithm
for episode in range(num_episodes):
    state = 0  # Start in state 1 (index 0)
    done = False

    while not done:
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(num_actions)  # Explore
        else:
            action = np.argmax(Q[state, :])  # Exploit

        # Take the action and observe the next state and reward
        if state == 0:
            next_state = 1 if action == 1 else 0 # Right goes to 2, Left stays in 1
        elif state == 1:
            next_state = 2 if action == 1 else 0 # Right goes to 3, Left goes to 1
        elif state == 2:
            next_state = 3 if action == 1 else 1 # Right goes to 4, Left goes to 2
        else:
            next_state = 3 # Stays in state 4

        reward = rewards[state, action]

        # Update the Q-value
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # Move to the next state
        state = next_state

        if state == 3:
            done = True # Episode ends when reaching state 4

print("Learned Q-table:")
print(Q)
