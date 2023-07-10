import math
import serial
import tensorflow as tf
import numpy as np
from collections import deque

'''
Initialization of number of features and actions
num_features - the number of features (or input dimensions) used to describe the state of the environment (the robot).
In our case, there are 3 features: speed, angle, and gyro. 
num_actions -  number of possible actions that the agent can take in the environment (the robot). 
In our case, there are 516 possible actions.
'''
num_features = 3
num_actions = 516


class DQN(tf.keras.Model):
    """Dense neural network class."""

    def __init__(self):
        """Initialization of layers of the neural network - three dense layers with ReLU activation functions"""
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32)  # No activation

    def call(self, x):
        """Forward pass."""
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)

'''
Creation of instances of DQN network - main neural network and target neural network
main_nn - responsible for making predictions and selecting actions based on the current state.
target_nn - used as a reference or a fixed target for estimating the target Q-values during the training process. Helps
stabilize and improve the training providing more consistent target values.
'''
main_nn = DQN()
target_nn = DQN()

'''
Initialization of Adam optimizer with learning rate of 1e-4 and mean square error loss function.
'''
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))  # adding experience to a buffer

    def __len__(self):
        return len(self.buffer)  # returns current size of buffer

    def sample(self, num_samples):
        """Sampling of batch of experiences from the buffer."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)  # Randomly select indices from the replay buffer.
        for i in idx:
            elem = self.buffer[i]  # Retrieve the experience tuple at the selected index.
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))  # Append the state to the states list (NumPy array).
            actions.append(np.array(action, copy=False))  # Append the action to the actions list (NumPy array).
            rewards.append(reward)  # Append the reward to the rewards list.
            next_states.append(np.array(next_state, copy=False))  # Append the next state to the next_states list (NumPy array).
            dones.append(done)   # Append the done flag to the dones list.

        # Converting to NumPy arrays
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones


def select_epsilon_greedy_action(state, epsilon):
    """Take random action with probability epsilon, else take best action."""
    result = tf.random.uniform((1,))
    if result < epsilon:
        return np.random.randint(-255, 256)  # Random action (left or right).
    else:
        return tf.argmax(main_nn(state)[0]).numpy()  # Greedy action for state.


@tf.function
def train_step(states, actions, rewards, next_states, dones):
    """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
    # Calculate targets.
    next_qs = target_nn(next_states)  # Forward pass through target network.
    max_next_qs = tf.reduce_max(next_qs, axis=-1)  # Select maximum Q-value.
    target = rewards + (1. - dones) * discount * max_next_qs  # Calculate target Q-value.
    with tf.GradientTape() as tape:
        qs = main_nn(states)  # Forward pass through main network.
        action_masks = tf.one_hot(actions, num_actions)  # Create one-hot action masks.
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)  # Mask and aggregate Q-values.
        loss = mse(target, masked_qs)  # Compute the mean squared error loss.
    grads = tape.gradient(loss, main_nn.trainable_variables)  # Compute gradients of the loss in relation to the trainable variables of the main_nn

    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))  # Update main network weights.
    return loss


def rad(angle):
    """Convert angles from degrees to radians"""
    return angle * math.pi / 180

# connectivity setup
uart_port = 'COM12'  # in case of UART connectivity
uart_speed = 115200  # serial port speed
wifi_local_ip = '192.168.4.1'  # host (this) IP
wifi_robot_ip = '192.158.4.2'  # remote (robot) IP
wifi_local_port = '1234'  # host (this) port
wifi_robot_port = '1235'  # remote (robot) port
ser = serial.Serial(
    port='COM12',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=0.01
)

# Learning hyperparameters.
num_episodes = 1000  # Number of episodes for training.
epsilon = 1.0   # Exploration rate (initial value).
batch_size = 32  # Batch size for training.
discount = 0.99  # Discount factor for future rewards.
buffer = ReplayBuffer(100000)  # Replay buffer to store experiences.
cur_frame = 0  # Current frame count.
msg = ''
last_100_ep_rewards = []  # List to store the rewards of the last 100 episodes.

for episode in range(num_episodes + 1):
    print("NEW EPISODE")
    # data from robot
    if episode == 1:
        msg = ser.readline()  # Read a line from the serial port for the first episode.
    while msg == b'PID\r\n' or msg == b'' or msg == '':
        msg = ser.readline()  # Read lines until a non-empty line with "PID" is received.

    msg = msg.split()
    angle = rad(float(msg[0][2:]))  # Extract angle from the received message and convert it to radians.
    speed = int(msg[1])  # Extract speed from the received message.
    gyro = float(msg[2])  # Extract gyro from the received message.

    state = [speed, angle, gyro]  # Create a state list

    ep_reward, done = 0, False  # Episode reward and done flag
    while not done:
        state_in = tf.expand_dims(state, axis=0)  # Expand dimensions of the state for NN.
        action = select_epsilon_greedy_action(state_in, epsilon)  # Select an action using epsilon-greedy policy.

        # send data to robot
        ser.write((str(action) + '\n').encode())  # Send action to robot.
        print("sent: " + str(action))
        msg = ser.readline()  # Response from robot.
        print(msg)
        if msg == b'PID\r\n':
            done = True  # If the message indicates the end of the episode, set the done flag to True.
        else:
            msg = msg.split()
            angle = rad(float(msg[0][2:]))  # Extract angle from message
            speed = int(msg[1])  # Extract speed from message
            gyro = float(msg[2])  # Extract gyro from message

        next_state = [speed, angle, gyro]  # Next state list
        # time based
        reward = 1

        ep_reward += reward  # Add the reward for the episode.
        # Save to experience replay.
        buffer.add(state, action, reward, next_state, done)  # Add the experience to the replay buffer.
        state = next_state  # Current state to next state
        cur_frame += 1
        # Copy main_nn weights to target_nn.
        if cur_frame % 2000 == 0:
            target_nn.set_weights(main_nn.get_weights())   # Update the target neural network weights.

        # Train neural network.
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)  # Sample a batch from the replay buffer.
            loss = train_step(states, actions, rewards, next_states, dones)  # Training

    if episode < 950:
        epsilon -= 0.001  # Decrease epsilon over time to decrease exploration.

    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]  # Remove the oldest episode reward from the list.
    last_100_ep_rewards.append(ep_reward)  # Add the current episode reward to the list.

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
              f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')  # Printing information about episode.

