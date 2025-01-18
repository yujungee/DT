import gym
import numpy as np
from metaworld import MT50
import random

ML10_TASK_MEAN = [ 0.02475038,  0.67194722,  0.16444747,  0.58618474,  0.03495118,  0.68820543,
                   0.12159905,  0.39969211, -0.03117681,  0.05358117,  0.45039071,  0.,
                   0.,          0.,          0.,          0.,          0.,          0.,
                   0.02389201,  0.67063919,  0.16474176,  0.58959416,  0.0315379,   0.68032548,
                   0.11530386,  0.39985029, -0.03150705,  0.05336399,  0.450496,    0.,
                   0.,          0.,          0.,          0.,          0.,          0.,
                   0.0506361,   0.75150876,  0.15012469]
ML10_TASK_STD = [  1.25007594e-01, 1.13586136e-01, 9.83395858e-02, 3.02065835e-01,
                   1.22944327e-01, 1.04849964e-01, 1.12581675e-01, 4.64258312e-01,
                   9.81644626e-02, 1.57954516e-01, 4.79230376e-01, 1.00000000e-06,
                   1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                   1.00000000e-06, 1.00000000e-06, 1.25288836e-01, 1.13613071e-01,
                   9.78864268e-02, 3.03138924e-01, 1.20882711e-01, 1.08384514e-01,
                   1.07948907e-01, 4.64178424e-01, 9.90633120e-02, 1.57385115e-01,
                   4.79081830e-01, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                   1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                   1.75725616e-01, 1.04983194e-01, 1.19016660e-01 ]

MT10_TASK = ['drawer-close-v2', 'reach-v2', 'window-close-v2', 'window-open-v2', 'button-press-topdown-v2',
             'door-open-v2', 'drawer-open-v2', 'pick-place-v2', 'peg-insert-side-v2', 'push-v2']
MT10_TASK_MEAN = [-0.00171678,  0.68572094,  0.16206354,  0.52179083, -0.00706913,  0.69211732,
                0.12525238,  0.21804859, -0.11712016,  0.04210936,  0.44833789,  0.,
                0.,          0.,          0.,          0.,          0.,          0.,
                -0.0015003,   0.68489412,  0.16234772,  0.52440989, -0.00177486,  0.69202918,
                0.12520434,  0.21825322, -0.11688009,  0.04231153,  0.44808569,  0.,
                0.,          0.,          0.,          0.,          0.,          0.,
                -0.03108508,  0.73052824,  0.14025199]

MT10_TASK_STD = [1.20266896e-01, 1.40102666e-01, 8.59032271e-02, 2.94791682e-01,
                1.33986192e-01, 1.26430245e-01, 7.44400865e-02, 3.53325233e-01,
                2.53179999e-01, 1.43772001e-01, 4.75763443e-01, 1.00000000e-06,
                1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                1.00000000e-06, 1.00000000e-06, 1.19839775e-01, 1.40231391e-01,
                8.58047734e-02, 2.96092898e-01, 1.27820006e-01, 1.26166352e-01,
                7.46057030e-02, 3.53487487e-01, 2.52884012e-01, 1.44420351e-01,
                4.75788828e-01, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                1.00000000e-06, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
                1.37624449e-01, 1.55019268e-01, 6.48557008e-02,]
MT50_TASK = [
    "assembly-v2", "basketball-v2", "bin-picking-v2", "box-close-v2", "button-press-topdown-v2", "button-press-topdown-wall-v2", "button-press-v2", "button-press-wall-v2", "coffee-button-v2", "coffee-pull-v2", 
    "coffee-push-v2", "dial-turn-v2", "disassemble-v2", "door-close-v2", "door-lock-v2", "door-open-v2", "door-unlock-v2", "hand-insert-v2", "drawer-close-v2", "drawer-open-v2", 
    "faucet-open-v2", "faucet-close-v2", "hammer-v2", "handle-press-side-v2", "handle-press-v2", "handle-pull-side-v2", "handle-pull-v2", "lever-pull-v2", "peg-insert-side-v2", "pick-place-wall-v2", 
    "reach-v2", "push-back-v2", "push-v2", "pick-place-v2", "plate-slide-v2", "plate-slide-side-v2", "plate-slide-back-v2", "plate-slide-back-side-v2", "peg-unplug-side-v2", "soccer-v2", 
    "stick-push-v2", "stick-pull-v2", "push-wall-v2", "reach-wall-v2", "shelf-place-v2", "sweep-into-v2", "sweep-v2", "window-open-v2", "window-close-v2", "pick-out-of-hole-v2",
]
MT50_TASK_MEAN = [ 0.0182423,   0.68267371,  0.1521013,   0.58392579,  0.01904824,  0.69695144,
  0.10564174,  0.40971446, -0.04293838,  0.06021696,  0.43973457,  0.01798605,
  0.03741483,  0.00763265,  0.02040816,  0.,          0.,          0.,
  0.01805152,  0.6818283,   0.15236583,  0.58623729,  0.01890903,  0.6941637,
  0.10347075,  0.40976184, -0.04296829,  0.06018971,  0.43977072,  0.01794383,
  0.03741927,  0.00763265,  0.02040816,  0.,          0.,          0.,
  0.03022892,  0.73047108,  0.11487805,]
MT50_TASK_STD = [1.27283328e-01, 1.19982370e-01, 8.91335251e-02, 2.94124037e-01,
 1.33460782e-01, 1.07451589e-01, 9.66369582e-02, 4.56906712e-01,
 1.74222349e-01, 1.86678294e-01, 4.65271156e-01, 7.36416114e-02,
 1.47971342e-01, 2.99987508e-02, 1.41392903e-01, 1.00000000e-06,
 1.00000000e-06, 1.00000000e-06, 1.27046727e-01, 1.20238219e-01,
 8.89713006e-02, 2.94935790e-01, 1.30644377e-01, 1.07828699e-01,
 9.45572696e-02, 4.56893928e-01, 1.74136275e-01, 1.86647293e-01,
 4.65253247e-01, 7.34780802e-02, 1.47976787e-01, 2.99987508e-02,
 1.41392903e-01, 1.00000000e-06, 1.00000000e-06, 1.00000000e-06,
 1.51097861e-01, 1.15503417e-01, 9.82216484e-02,]

class SingleTask(gym.Env):
    def __init__(self, task, seed=777, mode='eval', target_return=5000, target_timestep=500, index=(None,None)):
        print("Initialied by mj-metaworld, task:", task)
        random.seed(seed)
        self.mt = MT50(seed=seed)
        self.task_name = task
        self.mode = mode
        self.target_return = target_return
        self.target_timestep = target_timestep
        self._index = index

        if self.mode == 'eval':
            self._epi_idx = 0
        self.env_cls = self.mt.train_classes[self.task_name]
        self.env = self.env_cls()

        task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._return = None
        self._timestep = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.task_name == 'pick-place-v2' and self.mode == 'train':
            tcp = self.env.tcp_center
            obj = obs[4:7]
            reward -= np.clip(np.linalg.norm(obj - tcp) - 0.02, a_min=0, a_max=1e+4)
            reward -= np.clip(obj[2] - self.env.obj_init_pos[2] - 0.01, a_min=0, a_max=1e+4)
        
        self._return += reward
        self._timestep += 1
        info['task_name'] = self.task_name
        if info['success'] and self._return >= self.target_return:
            print("return", self._return)
            done = True
        if not done and self._timestep == self.target_timestep:
            done = True

        if self._index[0] is not None:
            context = np.zeros(self._index[1])
            context[self._index[0] % self._index[1]] = 1
            obs = np.concatenate([context, obs])
        
        return obs, reward, done, info

    def reset(self):
        self.env = self.env_cls()
        if self.mode == 'eval':
            self._epi_idx += 1
            task = [task for task in self.mt.train_tasks if task.env_name == self.task_name][self._epi_idx % 50]
        else:
            task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)
        
        obs = self.env.reset()
        if self._index[0] is not None:
            context = np.zeros(self._index[1])
            context[self._index[0] % self._index[1]] = 1
            obs = np.concatenate([context, obs])

        self._return = 0
        self._timestep = 1
        return obs

    def render(self, mode='human'):
        return self.env.render(offscreen=True, resolution=(224, 224))


class MetaWorldIndexedMultiTaskTester(gym.Env):
    def __init__(self, task_list, seed=777,):
        self.MTenv = MT50(seed=seed)
        self.env = None
        self.task_change = True

        self._task_idx = 0
        self._epi_idx = 0

        self.episode_length = episode_length
        self.mode = mode
        self.task_list = task_list

        env_cls = self.MTenv.train_classes[task_list[0]]
        env = env_cls()
        task = random.choice([task for task in self.MTenv.train_tasks if task.env_name == task_list[0]])
        env.set_task(task)

        self.observation_space = gym.spaces.Box(low=np.concatenate([env.observation_space.low, np.zeros(len(self.task_list))]),
                                            high=np.concatenate([env.observation_space.high, np.ones(len(self.task_list))]))

        self.action_space = env.action_space

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        context = np.zeros(len(self.task_list))
        context[self._task_idx % len(self.task_list)] = 1
        observation = np.concatenate([state, context])

        info['task_name'] = self.task_list[self._task_idx]
        info['task_idx'] = self._task_idx
        return observation, reward, done, info

    def set_task(self, task_num):
        self._task_idx = task_num
        self.task_change = False

    def reset(self):
        if self.task_change:
            self._epi_idx = (self._epi_idx + 1) % 500
            self._task_idx = self.epi_idx % len(self.task_list)
        else:
            self._epi_idx = (self._epi_idx + 10) % 500

        task_name = self.task_list[self._task_idx]
        env_cls = self.MTenv.train_classes[task_name]
        self.env = env_cls()
        tasks = [task for task in self.MTenv.train_tasks if task.env_name == task_name]
        self.env.set_task(tasks[self._epi_idx // 10])

        state = self.env.reset()
        context = np.zeros(len(self.task_list))
        context[self._task_idx] = 1
        observation = np.concatenate([state, context])
        return observation

    def render(self, mode='human'):
        return self.env.render(offscreen=True, resolution=(80, 80), camera_name='corner')

    @property
    def num_tasks(self):
        return len(self._train_task)


class TaskEnv(gym.Env):
    def __init__(self, task, seed=777, mode='eval', shuffle=0, max_timestep=500, index=(None,None)):
        print("Initialied by mj-metaworld, task:", task)
        random.seed(seed)
        self.mt = MT50(seed=seed)
        self.task_name = task
        self.mode = mode
        self.max_timestep = max_timestep
        self._index = index

        if self.mode == 'eval':
            self._epi_idx = 0
        else:
            random.seed(shuffle)
        self.env_cls = self.mt.train_classes[self.task_name]
        self.env = self.env_cls()

        task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)

        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self._return = None
        self._timestep = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.task_name == 'pick-place-v2' and self.mode == 'train':
            tcp = self.env.tcp_center
            obj = obs[4:7]
            reward -= np.clip(np.linalg.norm(obj - tcp) - 0.02, a_min=0, a_max=1e+4)
            reward -= np.clip(obj[2] - self.env.obj_init_pos[2] - 0.01, a_min=0, a_max=1e+4)
        
        self._return += reward
        self._timestep += 1
        info['task_name'] = self.task_name
        
        if not done and self._timestep == self.max_timestep:
            done = True

        if self._index[0] is not None:
            context = np.zeros(self._index[1])
            context[self._index[0] % self._index[1]] = 1
            obs = np.concatenate([context, obs])
        
        return obs, reward, done, info

    def reset(self):
        self.env = self.env_cls()
        if self.mode == 'eval':
            self._epi_idx += 1
            task = [task for task in self.mt.train_tasks if task.env_name == self.task_name][self._epi_idx % 50]
        else:
            task = random.choice([task for task in self.mt.train_tasks if task.env_name == self.task_name])
        self.env.set_task(task)
        
        obs = self.env.reset()
        if self._index[0] is not None:
            context = np.zeros(self._index[1])
            context[self._index[0] % self._index[1]] = 1
            obs = np.concatenate([context, obs])

        self._return = 0
        self._timestep = 1
        return obs
    
    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(offscreen=True, resolution=(224, 224))