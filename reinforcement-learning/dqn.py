import gym
import math
import sigopt
import numpy as np
import tensorflow as tf


# You can find your API token at https://sigopt.com/docs/overview/authentication
SIGOPT_API_KEY = 'YOUR_API_TOKEN_HERE'
# The game to train the DQN on. The default is CartPole-v0. You can also try LunarLander-v0 and Acrobot-v1
ENVIRONMENT_NAME = 'CartPole-v0'
# These values depend on which game you play. Choose reasonable values so that the agent has sufficient time to learn
MAX_EPISODES = 350
MAX_TIMESTEPS = 10000
# Determines whether the game should be rendered - set this to True to watch the agent play
SHOULD_RENDER = False


def main():
  conn = sigopt.Connection(client_token=SIGOPT_API_KEY)

  experiment = conn.experiments().create(
    name= ENVIRONMENT_NAME + ' (DQN)',
    observation_budget=128,
    parameters=[
      dict(name='hm', type='int', bounds=dict(min=5, max=100)),
      dict(name='mb_size', type='int', bounds=dict(min=10, max=500)),
      dict(name='e_decay', type='int', bounds=dict(min=MAX_EPISODES/10, max=MAX_EPISODES)),
      dict(name='log_lr', type='double', bounds=dict(min=math.log(0.00001), max=math.log(1.0))),
      dict(name='df', type='double', bounds=dict(min=0.5, max=0.9999)),
      dict(name='weight_sd', type='double', bounds=dict(min=0.01, max=0.5)),
      dict(name='bias_sd', type='double', bounds=dict(min=0.0, max=0.5))
    ]
  )

  # Run the SigOpt experiment loop
  # You can monitor the experiment's progress at https://sigopt.com/experiments
  for _ in range(experiment.observation_budget):
    suggestion = conn.experiments(experiment.id).suggestions().create()

    objective_metric = run_environment(
      hidden_multiplier=suggestion.assignments['hm'],
      minibatch_size=suggestion.assignments['mb_size'],
      epsilon_decay_steps=suggestion.assignments['e_decay'],
      learning_rate=math.exp(suggestion.assignments['log_lr']),
      discount_factor=suggestion.assignments['df'],
      initial_weight_stddev=suggestion.assignments['weight_sd'],
      initial_bias_stddev=suggestion.assignments['bias_sd']
    )

    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=objective_metric,
    )


def run_environment(
  hidden_multiplier,
  minibatch_size,
  epsilon_decay_steps,
  learning_rate,
  discount_factor,
  initial_weight_stddev,
  initial_bias_stddev
):

  env = gym.make(ENVIRONMENT_NAME)
  # The number of states about the environment the agent can observe
  observation_space_dim = env.observation_space.shape[0]
  # The number of actions the agent can make
  action_space_dim = env.action_space.n

  # The dimensions of the layers of the Q-network
  layer_dims = [
    observation_space_dim,
    hidden_multiplier * observation_space_dim,
    hidden_multiplier * observation_space_dim,
    action_space_dim
  ]

  neural_net = QNetwork(
    layer_dims=layer_dims,
    learning_rate=learning_rate,
    initial_weight_stddev=initial_weight_stddev,
    initial_bias_stddev=initial_bias_stddev
  )

  with tf.Session() as session:
    agent = Agent(
      session=session,
      neural_net=neural_net,
      action_space_dim=action_space_dim,
      minibatch_size=minibatch_size,
      discount_factor=discount_factor,
      epsilon_decay_steps=epsilon_decay_steps
    )

    session.run(tf.initialize_all_variables())

    # Run the environment feedback loop
    for episode in range(1, MAX_EPISODES + 1):
      observation = env.reset()
      reward = 0.0
      done = False
      action, done = agent.act(session, observation, reward, done, episode)

      while not done:
        if SHOULD_RENDER:
          env.render()
        observation, reward, done, _ = env.step(action)
        action, done = agent.act(session, observation, reward, done, episode)

  env.close()
  return agent.score()


class Agent:

  def __init__(
    self,
    session,
    neural_net,
    action_space_dim,
    minibatch_size,
    discount_factor,
    epsilon_decay_steps
  ):

    self._neural_net = neural_net
    self._action_space_dim = action_space_dim
    self._minibatch_size = minibatch_size
    self._discount_factor = discount_factor

    self._epsilon = 1.0
    self._final_epsilon = 0.1
    self._epsilon_decay = (self._epsilon - self._final_epsilon) / epsilon_decay_steps
    self._episodes_pure_exploration = 25

    self._num_rewards_to_average = 100
    self._replay_memory_size = 1000000

    self._timesteps = 0
    self._total_reward = 0.0
    self._rewards_list = []
    self._replay_buffer = []
    self._last_action = None
    self._last_state = None

    # Our objective metric
    self._best_average = None

  def score(self):
    return self._best_average

  def act(self, session, obs, reward, done, episode):

    self._total_reward += reward
    self._timesteps += 1

    # Putting a limit on how long the a single trial can run for simplicity
    if self._timesteps >= MAX_TIMESTEPS:
      done = True

    if done:
      # Record cumulative reward of trial
      self._rewards_list.append(self._total_reward)
      average = np.mean(self._rewards_list[-self._num_rewards_to_average:])

      if len(self._rewards_list) >= self._num_rewards_to_average and (self._best_average is None or self._best_average < average):
        self._best_average = average

      if self._epsilon > self._final_epsilon:
        self._epsilon -= self._epsilon_decay

      print ('Episode', episode, 'Reward', self._total_reward, 'Average Reward', round(average, 2))
      self._total_reward = 0.0
      self._timesteps = 0

    current_state = obs.reshape((1, len(obs)))

    # Initialize the last state and action
    if self._last_state is None:
      self._last_state = current_state
      q_values = self._neural_net.predict(session, self._last_state)
      best_action = np.argmax(q_values)
      self._last_action = np.zeros(self._action_space_dim)
      self._last_action[best_action] = 1
      return best_action, done

    # Store the current step
    new_step = [
      self._last_state.copy(),
      self._last_action.copy(),
      reward,
      current_state.copy(),
      done
    ]
    self._replay_buffer.append(new_step)
    self._last_state = current_state.copy()

    # If the buffer is full, remove the oldest step
    while len(self._replay_buffer) >= self._replay_memory_size:
      self._replay_buffer.pop(0)

    # Choose random actions during pure exploration
    if episode < self._episodes_pure_exploration:
      best_action = np.random.randint(0, self._action_space_dim)
    else:
      self._train(session)
      # Use epsilon-greedy policy to choose best action
      if np.random.random() > self._epsilon:
        q_values = self._neural_net.predict(session, self._last_state)
        best_action = np.argmax(q_values)
      else:
        best_action = np.random.randint(0, self._action_space_dim)

    next_action = np.zeros(self._action_space_dim)
    next_action[best_action] = 1
    self._last_action = next_action

    return best_action, done

  def _train(self, session):

    # Randomly sample from saved states
    permutations = np.random.permutation(len(self._replay_buffer))[:self._minibatch_size]
    minibatch = [self._replay_buffer[i] for i in permutations]

    previous_states = np.concatenate([step[0] for step in minibatch])
    actions = [step[1] for step in minibatch]
    rewards = np.array([step[2] for step in minibatch]).astype('float')
    current_states = np.concatenate([step[3] for step in minibatch])
    done = np.array([step[4] for step in minibatch]).astype('bool')

    # Current Q-values of the sampled states
    q_values = self._neural_net.predict(session, current_states)

    # Calculate target Q-values
    target_q_values = rewards.copy()
    target_q_values += ((1. - done) * self._discount_factor * q_values.max(axis=1))

    # Update the Q function with new information
    self._neural_net.fit(session, previous_states, actions, target_q_values)


class QNetwork:

  def __init__(self, layer_dims, learning_rate, initial_weight_stddev, initial_bias_stddev):
    self._input_layer = tf.placeholder("float", [None, layer_dims[0]])
    hidden_layers = [self._input_layer]

    weights = []
    biases = []

    for i in range(len(layer_dims) - 1):

      # Both weights and biases are initialized from a normal distribution
      weights.append(tf.Variable(tf.truncated_normal(
        [layer_dims[i], layer_dims[i+1]],
        stddev=initial_weight_stddev,
        mean=0.0
      )))

      biases.append(tf.Variable(tf.truncated_normal(
         shape=[layer_dims[i+1]],
         stddev=initial_bias_stddev,
         mean=0.0
      )))

      # Apply the sigmoid activation function to the hidden layers
      if i < len(layer_dims) - 2:
        hidden_layers.append(self._sigmoid(
          tf.matmul(hidden_layers[i], weights[i]) + biases[i]
        ))

    self._output_layer = tf.matmul(hidden_layers[-1], weights[-1])+ biases[-1]

    self._actions = tf.placeholder("float", [None, layer_dims[-1]])
    self._target_q_values = tf.placeholder("float", [None])

    # Calculating Q-values based on past states and actions
    self._predicted_q_values = tf.reduce_sum(
      tf.mul(self._output_layer, self._actions),
      reduction_indices=1
    )

    # Back propagation using TensorFlow's Adam Optimizer
    # and mean squared error as the cost function
    self._train_operation = tf.train.AdamOptimizer(learning_rate).minimize(
      self._mean_squared_error()
    )

  def _sigmoid(self, x):
    return 1.0/(1.0 + tf.exp(-x))

  def _mean_squared_error(self):
    return tf.reduce_mean(tf.square(self._target_q_values - self._predicted_q_values))

  def predict(self, session, state):
    return session.run(
      self._output_layer,
      feed_dict={self._input_layer: state}
    )

  def fit(self, session, states, actions, target_q_values):
    session.run(
      self._train_operation,
      feed_dict={
        self._input_layer: states,
        self._actions: actions,
        self._target_q_values: target_q_values
      }
    )


if __name__=="__main__":
   main()
