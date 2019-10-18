import gym

from PIL import Image
import numpy as np

from collections import deque
from gym import spaces



class SkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """
        Return only every `skip`-th frame (frameskipping)

        :param env: (Gym Environment) the environment
        :param skip: (int) number of `skip`-th frame
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,)+env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        """
        Step the environment with the given action
        Repeat action, sum reward, and max over last observations.

        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ChangeColorEnv(gym.Wrapper):

    def __init__(self, env, color_range):

        gym.Wrapper.__init__(self, env)
        self.color_range = color_range
        self.this_episode_color = [102,204,102]


    def reset(self, **kwargs):

        obs = self.env.reset(**kwargs)
        self.this_episode_color = np.random.randint(int(255*self.color_range[0]), int(255*self.color_range[1]), 3)

        obs_randomized = np.copy(obs)
        obs_randomized[np.where((obs_randomized[:,:,0]==102) & (obs_randomized[:,:,1]==204) & (obs_randomized[:,:,2]==102))] = self.this_episode_color

        return obs,obs_randomized

    def step(self,action):

        obs, rew, done, info = self.env.step(action)

        obs_randomized = np.copy(obs)
        obs_randomized[np.where((obs_randomized[:,:,0]==102) & (obs_randomized[:,:,1]==204) & (obs_randomized[:,:,2]==102))] = self.this_episode_color

        return obs,obs_randomized, rew, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, n_frames):
        """Stack n_frames last frames.

        Returns lazy array, which is much more memory efficient.

        See Also
        --------
        stable_baselines.common.atari_wrappers.LazyFrames

        :param env: (Gym Environment) the environment
        :param n_frames: (int) the number of frames to stack
        """
        gym.Wrapper.__init__(self, env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        self.frames_randomized = deque([], maxlen=n_frames)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * n_frames),
                                            dtype=env.observation_space.dtype)

    def reset(self):
        obs,obs_randomized = self.env.reset()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        for _ in range(self.n_frames):
            self.frames_randomized.append(obs_randomized)
        get_ob = self._get_ob()
        return get_ob[0], get_ob[1]

    def step(self, action):
        obs, obs_randomized, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        self.frames_randomized.append(obs_randomized)
        get_ob = self._get_ob()
        return get_ob[0], get_ob[1], reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n_frames
        return LazyFrames(list(self.frames)),LazyFrames(list(self.frames_randomized))

class LazyFrames(object):
    def __init__(self, frames):
        """
        This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.

        This object should only be converted to np.ndarray before being passed to the model.

        :param frames: ([int] or [float]) environment frames
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=2)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]

def make_carracing(env_id,color_range):

    env = gym.make(env_id)
    env = SkipEnv(env,skip=5)
    env = ChangeColorEnv(env,color_range)
    env = FrameStack(env,2)

    return env