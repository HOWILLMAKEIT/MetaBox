from torch import nn
from torch.distributions import Normal

from agents.networks import MultiHeadEncoder, MLP, EmbeddingNet
from basic_agent.PPO_Agent import *

class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class Actor(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 n_heads_actor,
                 n_layers,
                 normalization,
                 hidden_dim1,
                 hidden_dim2,
                 output_dim,
                 global_dim,
                 local_dim,
                 ind_dim
                 ):
        super(Actor, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_heads_actor = n_heads_actor     
        self.n_layers = n_layers
        self.normalization = normalization
        self.output_dim=output_dim
        self.global_dim = global_dim
        self.local_dim = local_dim
        self.ind_dim = ind_dim

        # figure out the Actor network
        # figure out the embedder for feature embedding
        self.embedder_1 = EmbeddingNet(
                            int(self.global_dim + self.local_dim),
                            int(self.embedding_dim / 2))
        self.embedder_2 = EmbeddingNet(
                            self.ind_dim,
                            int(self.embedding_dim / 2))
        # figure out the fully informed encoder
        self.encoder = mySequential(*(
                MultiHeadEncoder(self.n_heads_actor,
                                self.embedding_dim,
                                self.hidden_dim,
                                self.normalization, )
                for _ in range(self.n_layers)))  # stack L layers
        
        # figure out the mu_net and sigma_net
        mlp_config = [{'in': self.embedding_dim,'out': hidden_dim1,'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': self.output_dim,'drop_out':0,'activation':'None'}]
        self.decoder = MLP(mlp_config) 
        self.softmax = nn.Softmax(dim = -1)
        
        print(self.get_parameter_number())

    def get_parameter_number(self):
        
        total_num = sum(p.numel() for p in self.parameters())
        trainable_num = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    def forward(self, x_in, fixed_action = None, require_entropy = False, to_critic = False,only_critic=False, sampling = True):
        
        population_feature=x_in[:,:,:(self.global_dim + self.local_dim)]
        ind_feature=x_in[:,:,(self.global_dim + self.local_dim):]
        

        # pass through embedder
        h_em_1 = self.embedder_1(population_feature)
        h_em_2 = self.embedder_2(ind_feature)

        h_em = torch.cat((h_em_1, h_em_2), dim = -1)
        assert h_em.shape == (x_in.shape[0], x_in.shape[1], self.embedding_dim)

        # pass through encoder
        logits = self.encoder(h_em)
            
        # share logits to critic net, where logits is from the decoder output 
        if only_critic:
            return logits  # .view(bs, dim, ps, -1)
        
        probs = self.softmax(self.decoder(logits))

        # don't share the network between actor and critic if there is no attention mechanism
        _to_critic= logits

        policy = torch.distributions.Categorical(probs)
        

        if fixed_action is not None:
            action = torch.tensor(fixed_action)
        else:
            if sampling:
                action = policy.sample()
            else:
                action = torch.argmax(probs, dim = -1)
        assert action.shape == (x_in.shape[0], x_in.shape[1])
        # get log probability
        log_prob=policy.log_prob(action)

        # The log_prob of each instance is summed up, since it is a joint action for a population
        log_prob=torch.sum(log_prob,dim=1)

        
        if require_entropy:
            entropy = policy.entropy()  # for logging only 
            
            out = (action,
                   log_prob,
                   entropy,
                   _to_critic if to_critic else None,
                   )
        else:
            out = (action,
                   log_prob,
                   _to_critic if to_critic else None,
                   )
        return out

class Critic(nn.Module):
    def __init__(self,
             input_dim,
             hidden_dim1,
             hidden_dim2
             ):
        
        super(Critic, self).__init__()
        self.input_dim = input_dim
        # for GLEET, hidden_dim1 = 32, hidden_dim2 = 16
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        mlp_config = [{'in': self.input_dim,'out': hidden_dim1, 'drop_out': 0,'activation':'LeakyReLU'},
                  {'in': hidden_dim1,'out': hidden_dim2,'drop_out':0,'activation':'LeakyReLU'},
                  {'in': hidden_dim2,'out': 1,'drop_out':0,'activation':'None'}]
        self.value_head=MLP(config=mlp_config)

    def forward(self, h_features):
        # since it's joint actions, the input should be meaned at population-dimention
        h_features=torch.mean(h_features,dim=-2)
        # pass through value_head to get baseline_value
        baseline_value = self.value_head(h_features)
        
        return baseline_value.squeeze()


class RLEMMO_Agent(PPO_Agent):
    def __init__(self, config):
        self.config = config
        
        self.config.optimizer = 'Adam'
        self.config.lr_actor = 5e-4
        self.config.lr_critic = 5e-4
        self.config.lr_scheduler = 'ExponentialLR'
        
        # define parameters
        self.config.embedding_dim = 64
        self.config.encoder_head_num = 4
        self.config.n_encode_layers = 1
        self.config.normalization = 'layer'
        
        self.config.hidden_dim = 64
        self.config.hidden_dim1_actor = 16
        self.config.hidden_dim2_actor = 8
        self.config.hidden_dim1_critic = 16
        self.config.hidden_dim2_critic = 8
        self.output_dim = 5
        self.config.gamma = 0.99
        self.config.n_step = 10
        self.config.K_epochs = 3
        self.config.eps_clip = 0.1 
        self.config.lr_decay = 0.9862327
        self.config.max_grad_norm = 0.1
        
        self.global_dim = 5
        self.local_dim = 5
        self.ind_dim = 12

        
        # figure out the actor network
        actor = Actor(
            embedding_dim = self.config.embedding_dim,
            hidden_dim = self.config.hidden_dim,
            n_heads_actor = self.config.encoder_head_num,
            n_layers = self.config.n_encode_layers,
            normalization = self.config.normalization,
            hidden_dim1=self.config.hidden_dim1_actor,
            hidden_dim2=self.config.hidden_dim2_actor,
            output_dim = self.output_dim,
            global_dim = self.global_dim,
            local_dim = self.local_dim,
            ind_dim = self.ind_dim,
        )

        # figure out the critic network
        critic = Critic(
            input_dim = self.config.embedding_dim,
            hidden_dim1 = self.config.hidden_dim1_critic,
            hidden_dim2 = self.config.hidden_dim2_critic,
        )

        super().__init__(self.config, {'actor': actor, 'critic': critic}, [self.config.lr_actor, self.config.lr_critic])
        
    def __str__(self):
        return "RLEMMO"

    def train_episode(self, 
                      envs, 
                      para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                      asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                      num_cpus: Optional[Union[int, None]] = 1,
                      num_gpus: int = 0,
                      required_info = ['normalizer', 'gbest']):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        
        memory = Memory()

        # params for training
        gamma = self.gamma
        n_step = self.n_step
        
        K_epochs = self.K_epochs
        eps_clip = self.eps_clip
        
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass
        
        t = 0
        # initial_cost = obj
        _R = torch.zeros(len(env))
        _loss = []
        # sample trajectory
        while not env.all_done():
            t_s = t
            total_cost = 0
            entropy = []
            bl_val_detached = []
            bl_val = []

            # accumulate transition
            while t - t_s < n_step and not env.all_done():  
                
                memory.states.append(state.clone())
                action, log_lh, entro_p, _to_critic = self.actor(state, require_entropy = True, to_critic=True, sampling = True)
                
                memory.actions.append(action.clone() if isinstance(action, torch.Tensor) else copy.deepcopy(action))
                memory.logprobs.append(log_lh)
                
                entropy.append(entro_p.detach().cpu())

                baseline_val = self.critic(_to_critic)
                baseline_val_detached = baseline_val.detach()
                
                bl_val_detached.append(baseline_val_detached)
                bl_val.append(baseline_val)

                # state transient
                state, rewards, is_end, info = env.step(action.cpu().numpy().squeeze())
                memory.rewards.append(torch.FloatTensor(rewards).to(self.device))
                # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
                _R += rewards
                # store info

                # next
                t = t + 1

                try:
                    state = torch.FloatTensor(state).to(self.device)
                except:
                    pass
            
            # store info
            t_time = t - t_s
            total_cost = total_cost / t_time

            # begin update
            old_actions = torch.stack(memory.actions)
            try:
                old_states = torch.stack(memory.states).detach()  # .view(t_time, bs, ps, dim_f)
            except:
                pass
            # old_actions = all_actions.view(t_time, bs, ps, -1)
            old_logprobs = torch.stack(memory.logprobs).detach().view(-1)

            # Optimize PPO policy for K mini-epochs:
            old_value = None
            for _k in range(K_epochs):
                if _k == 0:
                    logprobs = memory.logprobs

                else:
                    # Evaluating old actions and values :
                    logprobs = []
                    entropy = []
                    bl_val_detached = []
                    bl_val = []

                    for tt in range(t_time):
                        # get new action_prob
                        _, log_p, entro_p, _to_critic = self.actor(old_states[tt], fixed_action = old_actions[tt],
                                                        require_entropy = True,# take same action
                                                        to_critic = True,
                                                        sampling = True)

                        logprobs.append(log_p)
                        entropy.append(entro_p.detach().cpu())

                        baseline_val = self.critic(_to_critic)
                        baseline_val_detached = baseline_val.detach()
                        
                        bl_val_detached.append(baseline_val_detached)
                        bl_val.append(baseline_val)

                logprobs = torch.stack(logprobs).view(-1)
                entropy = torch.stack(entropy).view(-1)
                bl_val_detached = torch.stack(bl_val_detached).view(-1)
                bl_val = torch.stack(bl_val).view(-1)

                # get traget value for critic
                Reward = []
                reward_reversed = memory.rewards[::-1]
                # get next value
                R = self.critic(self.actor(state, only_critic = True, sampling = True)).detach()

                for r in range(len(reward_reversed)):
                    R = R * gamma + reward_reversed[r]
                    Reward.append(R)
                # clip the target:
                Reward = torch.stack(Reward[::-1], 0)
                Reward = Reward.view(-1)
                
                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = Reward - bl_val_detached

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                reinforce_loss = -torch.min(surr1, surr2).mean()

                # define baseline loss
                if old_value is None:
                    baseline_loss = ((bl_val - Reward) ** 2).mean()
                    old_value = bl_val.detach()
                else:
                    vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                    v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                    baseline_loss = v_max.mean()

                # check K-L divergence (for logging only)
                approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
                approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0
                # calculate loss
                loss = baseline_loss + reinforce_loss

                # update gradient step
                self.optimizer.zero_grad()
                loss.backward()
                _loss.append(loss.item())
                # Clip gradient norm and get (clipped) gradient norms for logging
                # current_step = int(pre_step + t//n_step * K_epochs  + _k)
                grad_norms = clip_grad_norms(self.optimizer.param_groups, self.config.max_grad_norm)

                # perform gradient descent
                self.optimizer.step()
                self.learning_time += 1
                if self.learning_time >= (self.config.save_interval * self.cur_checkpoint):
                    save_class(self.config.agent_save_dir, 'checkpoint' + str(self.cur_checkpoint), self)
                    self.cur_checkpoint += 1

                if self.learning_time >= self.config.max_learning_step:
                    memory.clear_memory()
                    _Rs = _R.detach().numpy().tolist()
                    return_info = {'return': _Rs, 'loss': np.mean(_loss),'learn_steps': self.learning_time, }
                    for key in required_info:
                        return_info[key] = env.get_env_attr(key)
                    env.close()
                    return self.learning_time >= self.config.max_learning_step, return_info

            memory.clear_memory()
        
        is_train_ended = self.learning_time >= self.config.max_learning_step
        _Rs = _R.detach().numpy().tolist()
        return_info = {'return': _Rs, 'loss': np.mean(_loss),'learn_steps': self.learning_time,}
        for key in required_info:
            return_info[key] = env.get_env_attr(key)
        env.close()
        return is_train_ended, return_info

    def cal_pr_sr(self, env):
        raw_PR = torch.zeros((len(env), 5))
        raw_SR = torch.zeros((len(env), 5))
        for i in range(len(env)):
            solu = env.envs[i].optimizer.particles['current_position'].copy()
            accuracy = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
            total_pkn = env.envs[i].problem.nopt
            for acc_level in range(5):
                nfp, _ = env.envs[i].problem.how_many_goptima(solu, accuracy[acc_level])
                raw_PR[i][acc_level] = torch.FloatTensor(nfp / total_pkn)
                if nfp >= total_pkn:
                    raw_SR[i][acc_level] = torch.FloatTensor(1)
        return raw_PR, raw_SR


    def rollout_batch_episode(self,
                              envs,
                              seeds = None,
                              para_mode: Literal['dummy', 'subproc', 'ray', 'ray-subproc'] = 'dummy',
                              asynchronous: Literal[None, 'idle', 'restart', 'continue'] = None,
                              num_cpus: Optional[Union[int, None]] = 1,
                              num_gpus: int = 0,
                              required_info = {}):
        if self.device != 'cpu':
            num_gpus = max(num_gpus, 1)
        env = ParallelEnv(envs, para_mode, asynchronous, num_cpus, num_gpus)
        if seeds is not None:
            env.seed(seeds)
        state = env.reset()
        try:
            state = torch.FloatTensor(state).to(self.device)
        except:
            pass

        R = torch.zeros(len(env))
        gbest = torch.zeros(len(env))
        raw_PRs = []
        raw_SRs = []
        generation = 0
        # sample trajectory
        while not env.all_done():
            if generation % (env.envs[0].maxfes // env.envs[0].ps // 50) == 0:
                raw_PR, raw_SR = self.cal_pr_sr(env)
                raw_PRs.append(raw_PR)
                raw_SRs.append(raw_SR)

            with torch.no_grad():
                action, log_lh, entro_p = self.actor(state, sampling = False)
                
            # state transient
            state, rewards, is_end, info = env.step(action.cpu().numpy().squeeze())
            # print('step:{},max_reward:{}'.format(t,torch.max(rewards)))
            R += torch.FloatTensor(rewards).squeeze()
            # store info
            try:
                state = torch.FloatTensor(state).to(self.device)
            except:
                pass
            generation += 1
        for i in range(len(env)):
            gbest[i] = torch.FloatTensor(env.envs[i].optimizer.particles['gbest_val'])
        raw_PR, raw_SR = self.cal_pr_sr(env)
        raw_PRs.append(raw_PR)
        raw_SRs.append(raw_SR)

        while len(raw_PRs) < 51:
            raw_PRs.append(raw_PRs[-1])
        while len(raw_SRs) < 51:
            raw_SRs.append(raw_SRs[-1])
        assert len(raw_PRs) == 51 and len(raw_SRs) == 51

        raw_PRs = np.array(raw_PRs)
        raw_SRs = np.array(raw_SRs)


        results = {'return': R, 'gbest': gbest, 'pr_list': raw_PRs, 'sr_list': raw_SRs, 'pr': raw_PRs[-1], 'sr': raw_SRs[-1]}
        for key in required_info.keys():
            results[key] = env.get_env_attr(required_info[key])
        return results


