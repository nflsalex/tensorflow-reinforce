import random
import numpy as np
import tensorflow as tf
import itertools


TINY = 1e-8
class PolicyGradientREINFORCE(object):

  def __init__(self, session,
                     optimizer,
                     policy_network,
                     state_dim,
                     num_actions,
                     init_exp=0.5,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     summary_every=100):

    # tensorflow machinery
    self.session        = session
    self.optimizer      = optimizer
    self.summary_writer = summary_writer

    # model components
    self.policy_network = policy_network

    # training parameters
    self.state_dim       = state_dim
    self.num_actions     = num_actions
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param

    # Simon Wang
    self.step_size    = 10
    self._reg_coeff = 1e-5
    self._subsample_factor = 0.1
    self._backtrack_ratio = 0.8
    self._max_backtracks = 15

    # exploration parameters
    self.exploration  = init_exp
    self.init_exp     = init_exp
    self.final_exp    = final_exp
    self.anneal_steps = anneal_steps

    # counters
    self.train_iteration = 0

    # rollout buffer
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.old_dist_buffer = []

    # record reward history for normalization
    self.all_rewards = []
    self.max_reward_length = 1000000

    # create and initialize variables
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
    self.session.run(tf.initialize_variables(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = summary_every

  def resetModel(self):
    self.cleanUp()
    self.train_iteration = 0
    self.exploration     = self.init_exp
    var_lists = tf.get_collection(tf.GraphKeys.VARIABLES)
    self.session.run(tf.initialize_variables(var_lists))



  # def flatten_tensors(tensors):

  #   return np.concat(map(lambda x: np.reshape(x, [-1]), tensors))

  def unflatten_tensors(self, x):
    # tensor_sizes = map(np.prod, tensor_shapes)
    # indices = np.cumsum(tensor_sizes)[:-1]
    # return map(lambda pair: np.reshape(pair[0], pair[1]), zip(np.split(flattened, indices), tensor_shapes))
    var_tuple = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    start = 0
    xs = []
    for var in var_tuple:
      size = tf.shape(var)
      tmp1 = tf.slice(x, [start], [tf.size(var)])
      tmp = tf.reshape(tmp1, size)
      xs.append(tmp)
      start += tf.size(var)
    # assert start == tf.shape(x)[0] 
    xs = tuple(xs)
    return xs


  def cg(self, f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
      if callback is not None:
          callback(x)
      if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
      z = f_Ax(p)
      v = rdotr / p.dot(z)
      x += v * p
      r -= v * z
      newrdotr = r.dot(r)
      mu = newrdotr / rdotr
      p = r + mu * p

      rdotr = newrdotr
      if rdotr < residual_tol:
          break

    if callback is not None:
      callback(x)
    if verbose: print(fmtstr % (i + 1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x


  def create_variables(self):

    with tf.name_scope("model_inputs"):
      # raw state representation
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")

    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # initialize policy network
      with tf.variable_scope("policy_network"):
        self.policy_outputs = self.policy_network(self.states)

      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
      # Note 1: tf.multinomial is not good enough to use yet
      # so we don't use self.predicted_actions for now
      self.predicted_actions = tf.multinomial(self.action_scores, 1)

    # regularization loss
    policy_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_network")

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.taken_actions = tf.placeholder(tf.bool, (None, self.num_actions ), name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")
      self.old_dist = tf.placeholder(tf.float32, (None, self.num_actions), name="old_distribution")

      with tf.variable_scope("policy_network", reuse=True):
        self.logprobs = self.policy_network(self.states)

      self.kl = tf.reduce_sum(tf.mul(self.old_dist, (tf.log(self.old_dist + TINY) - tf.log(self.logprobs + TINY))), 1)
      mean_kl = tf.reduce_mean(self.kl)
      # N = tf.shape(self.logprobs)[0]
      lr = tf.div(tf.boolean_mask(self.logprobs, self.taken_actions) + TINY, tf.boolean_mask(self.old_dist, self.taken_actions) + TINY)
      self.surr_loss = - tf.reduce_mean(tf.mul(lr, self.discounted_rewards))

      self.constraint_term, self.constraint_value = mean_kl, self.step_size

      grads = self.optimizer.compute_gradients(self.surr_loss)
      self.flat_grads = tf.concat(0, [tf.reshape(grad, [-1]) for grad, var in grads ])

      # compute policy loss and regularization loss
      # self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logprobs, self.taken_actions)
      # self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
      # self.reg_loss           = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_network_variables])
      # self.loss               = self.pg_loss + self.reg_param * self.reg_loss

      # compute gradients
      constraint_grads, _ = zip(*self.optimizer.compute_gradients(self.constraint_term))
      # grads = self.optimizer.compute_gradients(self.surr_loss)



      # compute policy gradients
      # for i, (grad, var) in enumerate(self.gradients):
      #   if grad is not None:
      #     self.gradients[i] = (grad * self.discounted_rewards, var)

      # for grad, var in self.gradients:
      #   tf.histogram_summary(var.name, var)
      #   if grad is not None:
      #     tf.histogram_summary(var.name + '/gradients', grad)

      # emit summaries
      # tf.scalar_summary("policy_loss", self.pg_loss)
      # tf.scalar_summary("reg_loss", self.reg_loss)
      # tf.scalar_summary("total_loss", self.loss)
      tf.scalar_summary("surr_loss", self.surr_loss)

    # training update
    # with tf.name_scope("train_policy_network"):
    #   # apply gradients to update policy network
    #   self.train_op = self.optimizer.apply_gradients(self.gradients)


      self.x = tf.placeholder(tf.float32, shape =[None], name = "y_fy")
      # print(tf.shape(self.x))
      xs = self.unflatten_tensors(self.x)
      
      Hx_loss =  tf.reduce_sum([tf.reduce_sum(g * x) for g, x in itertools.izip(constraint_grads, xs)])
      Hx_plain_splits, var_splits = zip(*self.optimizer.compute_gradients(Hx_loss))
      self.Hx_plain = tf.concat(0, [tf.reshape(s, [-1]) for s in Hx_plain_splits])



    self.summarize = tf.merge_all_summaries()
    self.no_op = tf.no_op()

  def sampleAction(self, states):
    # TODO: use this code piece when tf.multinomial gets better
    # sample action from current policy
    # actions = self.session.run(self.predicted_actions, {self.states: states})[0]
    # return actions[0]

    # temporary workaround
    def softmax(y):
      """ simple helper function here that takes unnormalized logprobs """
      maxy = np.amax(y)
      e = np.exp(y - maxy)
      return e / np.sum(e)
    action_scores = self.session.run(self.action_scores, {self.states: states})[0]
    self.old_dist_buffer.append(action_scores)
    # epsilon-greedy exploration strategy
    if random.random() < self.exploration:
      return random.randint(0, self.num_actions-1)
    else:

      # if np.array(self.state_buffer).shape[0] > 5:
      #   print(self.session.run(self.action_scores, {self.states: np.array(self.state_buffer)}).shape)
      action_probs  = action_scores - 1e-5
      action = np.argmax(np.random.multinomial(1, action_probs))
      return action


  def Hx(self, x):
    plain = self.session.run([
        self.Hx_plain,
      ], {
        self.states:             np.array(self.subsample_state_buffer),
        self.x:                  x,
        self.taken_actions:      np.array(self.subsample_action_buffer),
        self.discounted_rewards: np.array(self.subsample_discounted_rewards_buffer),
        self.old_dist :          np.array(self.subsample_old_dist_buffer)
      })[0] + self._reg_coeff * x
    return plain


  def updateModel(self):

    N = len(self.reward_buffer)
    r = 0 # use discounted reward to approximate Q value

    # compute discounted future rewards
    self.discounted_rewards_buffer = np.zeros(N)
    for t in reversed(xrange(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      self.discounted_rewards_buffer[t] = r

    self.subsample_state_buffer = self.state_buffer
    self.subsample_action_buffer = self.action_buffer
    self.subsample_discounted_rewards_buffer = self.discounted_rewards_buffer
    self.subsample_old_dist_buffer = self.old_dist_buffer
    # reduce gradient variance by normalization
    # self.all_rewards += discounted_rewards.tolist()
    # self.all_rewards = self.all_rewards[:self.max_reward_length]
    # discounted_rewards -= np.mean(self.all_rewards)
    # discounted_rewards /= np.std(self.all_rewards)

    # whether to calculate summaries
    calculate_summaries = self.train_iteration % self.summary_every == 0 and self.summary_writer is not None
    flat_g, loss_before = self.session.run([
        self.flat_grads,
        self.surr_loss
      ], {
        self.states:             np.array(self.subsample_state_buffer),
        self.taken_actions:      np.array(self.subsample_action_buffer),
        self.discounted_rewards: np.array(self.subsample_discounted_rewards_buffer),
        self.old_dist :          np.array(self.subsample_old_dist_buffer)
      })
    print(flat_g)
    print(loss_before)
    descent_direction = self.cg(self.Hx, flat_g)
    print(descent_direction)
    
    initial_step_size = np.sqrt(2.0 * self.step_size * (1. / (descent_direction.dot(self.Hx(descent_direction)) + 1e-8)))
    if np.isnan(initial_step_size):
      initial_step_size = 1.
    flat_descent_step = initial_step_size * descent_direction
    print "inital step size %f" % initial_step_size

    # print descent_direction
    # print 2.0 * self.step_size * (1. / (descent_direction.dot(self.Hx(descent_direction)) + 1e-8))
    # print initial_step_size

    # update policy network with the rollout in batches
    # for t in xrange(N-1):

    #   # prepare inputs
    #   states  = self.state_buffer[t][np.newaxis, :]
    #   actions = np.array([self.action_buffer[t]])
    #   rewards = np.array([discounted_rewards[t]])

    #   # evaluate gradients
    #   grad_evals = [grad for grad, var in self.gradients]

    #   # perform one update of training
    #   _, summary_str = self.session.run([
    #     self.train_op,
    #     self.summarize if calculate_summaries else self.no_op
    #   ], {
    #     self.states:             states,
    #     self.taken_actions:      actions,
    #     self.discounted_rewards: rewards
    #   })

    #   # emit summaries
    #   if calculate_summaries:
    #     self.summary_writer.add_summary(summary_str, self.train_iteration)
    print("loss_before: %f" % loss_before)

    prev_param = tf.concat(0, [tf.reshape(tf.identity(x), [-1]) for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])

    for n_iter, ratio in enumerate(self._backtrack_ratio ** np.arange(self._max_backtracks)):

      cur_step = ratio * flat_descent_step
      cur_param = prev_param - cur_step

      cur_param_tuple = self.unflatten_tensors(cur_param)
      for var, cur_var in  itertools.izip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), cur_param_tuple):
        reassign = var.assign(cur_var)
        self.session.run(reassign)
      loss, constraint_val, tmp = self.session.run([
        self.surr_loss,
        self.constraint_term,
        self.logprobs
      ], {
        self.states:             np.array(self.subsample_state_buffer),
        self.taken_actions:      np.array(self.subsample_action_buffer),
        self.discounted_rewards: np.array(self.subsample_discounted_rewards_buffer),
        self.old_dist :          np.array(self.subsample_old_dist_buffer)
      })
      print("loss: %f constraint: %f  n_iter: %d" % (loss, constraint_val, n_iter))
      # print(self.subsample_old_dist_buffer)
      # print(tmp)
      # print("haha")
      if loss < loss_before and constraint_val <= self.step_size:
        break
      
    if (np.isnan(loss) or np.isnan(constraint_val) or loss >= loss_before or constraint_val >=
        self.step_size):
      print("Line search condition violated. Rejecting the step!!!!!!!!!!!!")
      prev_param_tuple = self.unflatten_tensors(prev_param)
      for var, prev_var in  itertools.izip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), prev_param_tuple):
        reassign = var.assign(prev_var)
        self.session.run(reassign)

    print("backtrack iters: %d" % n_iter)
    
    print("loss: %f" % loss)


    self.annealExploration()
    self.train_iteration += 1

    # clean up
    self.cleanUp()

  def annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp
    # self.exploration = 0

  def storeRollout(self, state, action, reward):

    onehot_action = np.zeros((1, self.num_actions))
    onehot_action[0, action] = 1
    self.action_buffer.extend(onehot_action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)

  def cleanUp(self):
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []
    self.old_dist_buffer = []