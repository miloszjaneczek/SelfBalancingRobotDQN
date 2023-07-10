import math
import serial
import tensorflow as tf
import numpy as np
from collections import deque

num_features = 3
num_actions = 516


class DQN(tf.keras.Model):
    """Dense neural network class."""

    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(32, activation="relu")
        self.dense2 = tf.keras.layers.Dense(32, activation="relu")
        self.dense3 = tf.keras.layers.Dense(num_actions, dtype=tf.float32)  # No activation

    def call(self, x):
        """Forward pass."""
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


main_nn = DQN()
target_nn = DQN()

optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
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
    next_qs = target_nn(next_states)
    max_next_qs = tf.reduce_max(next_qs, axis=-1)
    target = rewards + (1. - dones) * discount * max_next_qs
    with tf.GradientTape() as tape:
        qs = main_nn(states)
        action_masks = tf.one_hot(actions, num_actions)
        masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
        loss = mse(target, masked_qs)
    grads = tape.gradient(loss, main_nn.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_nn.trainable_variables))
    return loss


def rad(angle):
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

# Learning
# Hyperparameters.
num_episodes = 1000
epsilon = 1.0
batch_size = 32
discount = 0.99
buffer = ReplayBuffer(100000)
cur_frame = 0
msg = ''
last_100_ep_rewards = []
for episode in range(num_episodes + 1):
    print("NEW EPISODE")
    # data from robot
    if episode == 1:
        msg = ser.readline()
    while msg == b'PID\r\n' or msg == b'' or msg == '':
        msg = ser.readline()

    msg = msg.split()
    angle = rad(float(msg[0][2:]))
    speed = int(msg[1])
    gyro = float(msg[2])

    state = [speed, angle, gyro]

    ep_reward, done = 0, False
    while not done:
        state_in = tf.expand_dims(state, axis=0)
        action = select_epsilon_greedy_action(state_in, epsilon)

        # send data to robot
        ser.write((str(action) + '\n').encode())
        print("sent: " + str(action))
        msg = ser.readline()
        print(msg)
        if msg == b'PID\r\n':
            done = True
        else:
            msg = msg.split()
            angle = rad(float(msg[0][2:]))
            speed = int(msg[1])
            gyro = float(msg[2])

        next_state = [speed, angle, gyro]
        # time based
        reward = 1

        ep_reward += reward
        # Save to experience replay.
        buffer.add(state, action, reward, next_state, done)
        state = next_state
        cur_frame += 1
        # Copy main_nn weights to target_nn.
        if cur_frame % 2000 == 0:
            target_nn.set_weights(main_nn.get_weights())

        # Train neural network.
        if len(buffer) >= batch_size:
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            loss = train_step(states, actions, rewards, next_states, dones)

    if episode < 950:
        epsilon -= 0.001

    if len(last_100_ep_rewards) == 100:
        last_100_ep_rewards = last_100_ep_rewards[1:]
    last_100_ep_rewards.append(ep_reward)

    if episode % 50 == 0:
        print(f'Episode {episode}/{num_episodes}. Epsilon: {epsilon:.3f}. '
              f'Reward in last 100 episodes: {np.mean(last_100_ep_rewards):.3f}')
