import numpy as np
import torch
import copy
from agent.basic_agent import Basic_Agent
from agent.networks import MLP
from agents.utils import *
from .DDQN_Agent import *
import math
class Surr_RLDE_Agent(DDQN_Agent):
	def __init__(self, config):
		self.config = config
		self.config.state_size = 9
		self.config.n_act = 15
		self.config.lr_model = 1e-4
		self.config.lr_decay = 1
		self.config.batch_size = 64
		self.config.epsilon = 0.5  # 0.5 - 0.05
		self.config.gamma = 0.99
		self.config.target_update_interval = 1000
		self.config.memory_size = 100000
		self.config.warm_up_size = config.batch_size
		self.config.net_config = [{'in': config.state_size, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': 64, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 64, 'out': 32, 'drop_out': 0, 'activation': 'ReLU'},
							 {'in': 32, 'out': config.n_act, 'drop_out': 0, 'activation': 'None'}]
		self.device = config.device

		self.config.max_grad_norm = math.inf
		# self.pred_Qnet = MLP(config.net_config).to(self.device)
		# self.target_Qnet = copy.deepcopy(self.pred_Qnet).to(self.device)
		# self.optimizer = torch.optim.AdamW(self.pred_Qnet.parameters(), lr=config.lr)
		# self.criterion = torch.nn.MSELoss()

		self.n_act = config.n_act
		self.epsilon = config.epsilon
		self.gamma = config.gamma
		# self.update_target_steps = config.update_target_steps
		# self.batch_size = config.batch_size
		# self.replay_buffer = ReplayBuffer(config.memory_size, device=self.device, state_dim=self.config.state_size)
		# self.warm_up_size = config.warm_up_size
		self.max_learning_step = config.max_learning_step

		self.cur_checkpoint = 0

		self.config.optimizer = 'AdamW'
		# origin code does not have lr_scheduler
		self.config.lr_scheduler = 'ExponentialLR'
		self.config.criterion = 'MSELoss'
		model = MLP(self.config.net_config).to(self.config.device)


		# save_class(self.config.agent_save_dir, 'checkpoint0', self)  # save the model of initialized agent.
		# self.learned_time = 0  # record the number of accumulated learned steps
		# self.learned_steps_history = 0

		# self.cur_checkpoint += 1  # record the current checkpoint of saving agent
		super().__init__(self.config, {'model': model}, self.config.lr_model)
	# def get_action(self, state, options=None):
	# 	"""
	# 	Parameter
	# 	----------
	# 	state: state features defined by developer.
	#
	# 	Return
	# 	----------
	# 	action: the action inferenced by using state.
	# 	"""
	# 	state = torch.tensor(state).to(self.device)
	# 	action = None
	#
	# 	with torch.no_grad():
	# 		Q_list = self.pred_Qnet(state)
	# 	if options['epsilon_greedy'] and torch.rand(1) < self.epsilon:
	# 		action = torch.randint(0, self.n_act, (1,)).item()
	# 	if action is None:
	# 		action = torch.argmax(Q_list).item()
	# 	return action

	def get_epsilon(self, step, start=0.5, end=0.05):
		total_steps = self.config.max_learning_step
		return max(end, start - (start - end) * (step / total_steps))

	def get_action(self, state, epsilon_greedy=False):
		state = torch.Tensor(state).to(self.device)
		self.epsilon = self.get_epsilon(self.learning_time)
		with torch.no_grad():
			Q_list = self.model(state)
		if epsilon_greedy and np.random.rand() < self.epsilon:
			action = np.random.randint(low=0, high=self.n_act, size=len(state))
		else:
			action = torch.argmax(Q_list, -1).detach().cpu().numpy()
		return action

	# def train_episode(self,
	# 				  envs,
	# 				  para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
	# 				  asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
	# 				  num_cpus: Optional[Union[int, None]] = 1,
	# 				  num_gpus: int = 0,
	# 				  required_info=['normalizer', 'gbest']):
	#
	# 	if self.device != 'cpu':
	# 		num_gpus = max(num_gpus, 1)
	# 	env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
	#
	# 	# params for training
	# 	gamma = self.gamma
	#
	# 	state = env.reset()
	# 	try:
	# 		state = torch.FloatTensor(state)
	# 	except:
	# 		pass
	#
	# 	_R = torch.zeros(len(env))
	# 	_loss = []
	# 	# sample trajectory
	# 	while not env.all_done():
	# 		action = self.get_action(state=state, epsilon_greedy=True)
	#
	# 		# state transient
	# 		next_state, reward, is_end, info = env.step(action)
	# 		_R += reward
	# 		# store info
	# 		# convert next_state into tensor
	# 		try:
	# 			next_state = torch.FloatTensor(next_state).to(self.device)
	# 		except:
	# 			pass
	# 		for s, a, r, ns, d in zip(state, action, reward, next_state, is_end):
	# 			self.replay_buffer.append((s, a, r, ns, d))
	# 		try:
	# 			state = next_state
	# 		except:
	# 			state = copy.deepcopy(next_state)
	#
	# 		# begin update
	# 		if len(self.replay_buffer) >= self.warm_up_size:
	# 			batch_obs, batch_action, batch_reward, batch_next_obs, batch_done = self.replay_buffer.sample(
	# 				self.batch_size)
	# 			pred_Vs = self.model(batch_obs.to(self.device))  # [batch_size, n_act]
	# 			action_onehot = torch.nn.functional.one_hot(batch_action.to(self.device),
	# 														self.n_act)  # [batch_size, n_act]
	#
	# 			predict_Q = (pred_Vs * action_onehot).sum(1)  # [batch_size]
	# 			target_Q = batch_reward.to(self.device) + (1 - batch_done.to(self.device)) * gamma * \
	# 					   self.target_model(batch_next_obs.to(self.device)).max(1)[0].detach()
	#
	# 			self.optimizer.zero_grad()
	# 			loss = self.criterion(predict_Q, target_Q)
	# 			loss.backward()
	# 			grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)
	# 			self.optimizer.step()
	#
	# 			_loss.append(loss.item())
	# 			self.learning_time += 1
	# 			if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
	# 				save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
	# 				self.cur_checkpoint += 1
	#
	# 			if self.learning_time % self.target_update_interval == 0:
	# 				for target_parma, parma in zip(self.target_model.parameters(), self.model.parameters()):
	# 					target_parma.data.copy_(parma.data)
	#
	# 			if self.learning_time >= self.config.max_learning_step:
	# 				_Rs = _R.detach().numpy().tolist()
	# 				return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
	# 				for key in required_info:
	# 					return_info[key] = env.get_env_attr(key)
	# 				env.close()
	# 				return self.learning_time >= self.config.max_learning_step, return_info
	#
	# 	is_train_ended = self.learning_time >= self.config.max_learning_step
	# 	_Rs = _R.detach().numpy().tolist()
	# 	return_info = {'return': _Rs, 'loss': np.mean(_loss), 'learn_steps': self.learning_time, }
	# 	for key in required_info:
	# 		return_info[key] = env.get_env_attr(key)
	# 	env.close()
	#
	# 	return is_train_ended, return_info
	#


	# def train_episode(self, env):
	# 	""" Called by Trainer.
	# 		Optimize a problem instance in training set until reaching max_learning_step or satisfy the convergence condition.
	# 		During every train_episode,you need to train your own network.
	#
	# 	Parameter
	# 	----------
	# 	env: an environment consisting of a backbone optimizer and a problem sampled from train set.
	#
	# 	Must To Do
	# 	----------
	# 	1. record total reward
	# 	2. record current learning steps and check if reach max_learning_step
	# 	3. save agent model if checkpoint arrives
	#
	# 	Return
	# 	----------
	# 	A boolean that is true when fes reaches max_learning_step otherwise false
	# 	A dict: {'normalizer': float,
	# 			 'gbest': float,
	# 			 'return': float,
	# 			 'learn_steps': int
	# 			 }
	# 	"""
	#
	# 	if self.learned_steps == 3136 or self.learned_steps - 3200 == self.learned_steps_history:
	# 		self.learned_steps_history = self.learned_steps
	# 		self.epsilon = self.epsilon - (0.5 - 0.05) / 468
	# 		# print(self.epsilon)
	#
	# 	state = env.reset()
	# 	R = 0  # total reward
	# 	is_done = False
	#
	# 	while not is_done:
	# 		action = self.get_action(state, {'epsilon_greedy': True})
	# 		next_state, reward, is_done = env.step(action)
	# 		R += reward
	# 		self.replay_buffer.append(state, action, reward, next_state, is_done)
	# 		if len(self.replay_buffer) > self.warm_up_size:
	# 			batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.replay_buffer.sample(self.batch_size)
	# 			pred_Q = (self.pred_Qnet(batch_state)).gather(1, batch_action.unsqueeze(-1)).squeeze(-1)
	#
	# 			# Target Q values
	# 			with torch.no_grad():
	# 				max_actions = self.pred_Qnet(batch_next_state).argmax(dim=1)
	# 				target_q_values = self.target_Qnet(batch_next_state).gather(1, max_actions.unsqueeze(-1)).squeeze(-1)
	# 				targets = batch_reward + (self.gamma * target_q_values * (1 - batch_done))
	#
	# 			# Loss and optimization
	# 			loss = self.criterion(pred_Q, targets)
	# 			self.optimizer.zero_grad()
	# 			loss.backward()
	# 			self.optimizer.step()
	# 			self.learned_steps += 1
	# 			# save agent model if checkpoint arrives
	# 			if self.learned_steps >= (self.config.save_interval * self.cur_checkpoint):
	# 				save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
	# 				self.cur_checkpoint += 1
	#
	# 			if self.learned_steps >= self.max_learning_step:
	# 				break
	#
	# 		state = next_state
	# 		if self.learned_steps % self.update_target_steps == 0:
	# 			for target_parma, parma in zip(self.target_Qnet.parameters(), self.pred_Qnet.parameters()):
	# 				target_parma.data.copy_(parma.data)
	#
	# 	return self.learned_steps >= self.config.max_learning_step, {'normalizer': env.optimizer.cost[0],
	# 																 'gbest': env.optimizer.cost[-1],
	# 																 'return': R,
	# 																 'learn_steps': self.learned_steps}
	#
	# def rollout_episode(self, env):
	# 	""" Called by method rollout and Tester.test
	#
	# 	Parameter
	# 	----------
	# 	env: an environment consisting of a backbone optimizer and a problem sampled from test set
	#
	# 	Return
	# 	----------
	# 	A dict: {'cost': list,
	# 	'fes': int,
	# 	'return': float
	# 	}
	# 	"""
	# 	env.optimizer.is_train = False
	# 	state = env.reset()
	# 	is_done = False
	# 	R = 0  # total reward
	# 	while not is_done:
	# 		action = self.get_action(state, {'epsilon_greedy': False})
	# 		next_state, reward, is_done = env.step(action)  # feed the action to environment
	# 		R += reward  # accumulate reward
	# 		state = next_state
	#
	# 	return {'cost': env.optimizer.cost, 'fes': env.optimizer.fes, 'return': R}