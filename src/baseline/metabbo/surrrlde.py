from .networks import MLP
from rl.ddqn import *


class SurrRLDE(DDQN_Agent):
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

		super().__init__(self.config, {'model': model}, self.config.lr_model)


	def __str__(self):
		return "Surr_RLDE"

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
