from __future__ import division
from collections import deque
from copy import deepcopy

import numpy as np
import keras.backend as K
from keras.models import Model

from rl.core import Agent
from rl.util import *
from rl.agents.props_util import *
from scipy.optimize import minimize

class CEMAgent(Agent):
    """Write me
    """
    def __init__(self, model, nb_actions, memory, batch_size=50, nb_steps_warmup=1000,
                 train_interval=50, elite_frac=0.05, memory_interval=1, theta_init=None,
                 noise_decay_const=0.0, noise_ampl=0.0, delta=0.05, Lmax=10, bound_opts={}, **kwargs):
        super(CEMAgent, self).__init__(**kwargs)

        # Parameters.
        self.nb_actions = nb_actions
        self.batch_size = batch_size
        self.elite_frac = elite_frac
        self.num_best = int(self.batch_size * self.elite_frac)
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        
        # if using noisy CEM, the minimum standard deviation will be ampl * exp (- decay_const * step )
        self.noise_decay_const = noise_decay_const
        self.noise_ampl = noise_ampl
                
        # default initial mean & cov, override this by passing an theta_init argument
        self.init_mean = 0.0
        self.init_stdev = 1.0
        
        # Related objects.
        self.memory = memory
        self.model = model
        self.shapes = [w.shape for w in model.get_weights()]
        self.sizes = [w.size for w in model.get_weights()]
        self.num_weights = sum(self.sizes)
        
        # store the best result seen during training, as a tuple (reward, flat_weights)
        self.best_seen = (-np.inf, np.zeros(self.num_weights))

        self.theta = np.zeros(self.num_weights*2)
        self.update_theta(theta_init)

        # State.
        self.episode = 0
        self.compiled = False
        self.reset_states()

        # bound stuff
        self.delta = delta
        self.Lmax = Lmax
        self.bound_opts = bound_opts
        self.curr_th_mean = np.zeros(self.num_weights)
        self.curr_th_std = np.ones_like(self.curr_th_mean) * self.init_stdev
        self.pk0 = NormalDist(self.curr_th_mean, np.diag(np.power(self.curr_th_std, 2)))
        self.curr_pk = self.pk0
        self.pks = [self.pk0]
        self.yss = None
        self.thss = None
        self.bound_vals = [0]
        self.a = 1
        self.tol = 1e-20
        self.min_var = 1e-3;
        self.d = self.curr_th_mean.size

    def compile(self):
        self.model.compile(optimizer='sgd', loss='mse')
        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def get_weights_flat(self,weights):
        weights_flat = np.zeros(self.num_weights)

        pos = 0
        for i_layer, size in enumerate(self.sizes):
            weights_flat[pos:pos+size] = weights[i_layer].flatten()
            pos += size
        return weights_flat
        
    def get_weights_list(self,weights_flat):
        weights = []
        pos = 0
        for i_layer, size in enumerate(self.sizes):
            arr = weights_flat[pos:pos+size].reshape(self.shapes[i_layer])
            weights.append(arr)
            pos += size
        return weights          

    def reset_states(self):
        self.recent_observation = None
        self.recent_action = None

    def select_action(self, state, stochastic=False):
        batch = np.array([state])
        if self.processor is not None:
            batch = self.processor.process_state_batch(batch)

        action = self.model.predict_on_batch(batch).flatten()
        if stochastic or self.training:
            return np.random.choice(np.arange(self.nb_actions), p=np.exp(action) / np.sum(np.exp(action)))
        return np.argmax(action)
    
    def update_theta(self,theta):
        if (theta is not None):
            assert theta.shape == self.theta.shape, "Invalid theta, shape is {0} but should be {1}".format(theta.shape,self.theta.shape)
            assert (not np.isnan(theta).any()), "Invalid theta, NaN encountered"
            assert (theta[self.num_weights:] >= 0.).all(), "Invalid theta, standard deviations must be nonnegative"            
            self.theta = theta
        else:
            means = np.ones(self.num_weights) * self.init_mean
            stdevs = np.ones(self.num_weights) * self.init_stdev
            self.theta = np.hstack((means,stdevs))

    def choose_weights(self):
        mean = self.theta[:self.num_weights]
        std = self.theta[self.num_weights:]
        weights_flat = std * np.random.randn(self.num_weights) + mean

        sampled_weights = self.get_weights_list(weights_flat)
        self.model.set_weights(sampled_weights)

    def forward(self, observation):
        # Select an action.
        state = self.memory.get_recent_state(observation)
        action = self.select_action(state)
        if self.processor is not None:
            action = self.processor.process_action(action)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action

    @property
    def layers(self):
        return self.model.layers[:]
         
    def backward(self, reward, terminal):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        if terminal:
            params = self.get_weights_flat(self.model.get_weights())
            self.memory.finalize_episode(params)
            self.bound_vals.append(self.bound_vals[-1])

            if self.step > self.nb_steps_warmup and self.episode % self.train_interval == 0:
                params, reward_totals = self.memory.sample(self.batch_size)

                # bound stuff
                ths = np.array(params)
                ys = np.array(reward_totals)
		ys_trans = 200 - np.array([ys])
                ths_trans = np.array([ths]).transpose(2, 1, 0)

                if self.yss is None:
                    self.yss = ys_trans
                else:
                    self.yss = np.append(self.yss, ys_trans, axis=0)

                if self.thss is None:
                    self.thss = ths_trans
                else:
                    self.thss = np.append(self.thss, ths_trans, axis=2)

                # Setup box constraints on optimization vector        
                box_constraints = [(self.tol, None)] # alpha > 0
                    
                # CEM actual stuff
                best_idx = np.argsort(np.array(reward_totals))[-self.num_best:]
                best = np.vstack([params[i] for i in best_idx])

                if reward_totals[best_idx[-1]] > self.best_seen[0]:
                    self.best_seen = (reward_totals[best_idx[-1]], params[best_idx[-1]])
                    
                metrics = [np.mean(np.array(reward_totals)[best_idx])]
                if self.processor is not None:
                    metrics += self.processor.metrics
                min_std = self.noise_ampl * np.exp(-self.step * self.noise_decay_const)
                
                mean = np.mean(best, axis=0)
                std = np.std(best, axis=0) + min_std
                new_theta = np.hstack((mean, std))
                self.update_theta(new_theta)

                # calculate bound stuff
                analytic_jac = self.bound_opts.get('analytic_jac')
                a0 = self.a
                bound = lambda a : self.bound_util(a)
                res = minimize(bound, a0, method='L-BFGS-B', jac=analytic_jac, bounds=box_constraints, options={'disp' : False})

                self.a = res.x
                self.bound_vals.append(-1*res.fun + 200)

                self.curr_th_mean = mean
                self.curr_th_std = std
                self.curr_pk = NormalDist(self.curr_th_mean, np.diag(np.power(self.curr_th_std, 2)))
                self.pks.append(self.curr_pk)
            self.choose_weights()
            self.episode += 1
        return metrics

    def _on_train_end(self):
        self.model.set_weights(self.get_weights_list(self.best_seen[1]))

    @property
    def metrics_names(self):
        names = ['mean_best_reward']
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    def bound_util(self, a):
        val, _, _ = dist_bound_robust(a, self.curr_pk, self.pks, self.yss, self.thss, self.delta, self.Lmax, self.bound_opts)
        return val
