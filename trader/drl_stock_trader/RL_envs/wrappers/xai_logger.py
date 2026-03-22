import os
import json
import numpy as np
import gym

class XAILoggerWrapper(gym.Wrapper):
    """
    Logs (obs, action, reward, done) tuples for later surrogate training.
    Writes a compact NPZ at episode end.
    """
    def __init__(self, env, log_path: str):
        super().__init__(env)
        self.log_path = log_path
        self._obs = None
        self._buf_obs = []
        self._buf_act = []
        self._buf_rew = []
        self._buf_done = []

        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        return obs

    def step(self, action):
        # action can be np array or list; store as float32 vector
        act = np.asarray(action, dtype=np.float32).reshape(-1)

        next_obs, reward, done, info = self.env.step(action)
        obs_vec = np.asarray(self._obs, dtype=np.float32).reshape(-1)

        self._buf_obs.append(obs_vec)
        self._buf_act.append(act)
        self._buf_rew.append(float(reward))
        self._buf_done.append(bool(done))

        self._obs = np.asarray(next_obs, dtype=np.float32).reshape(-1)

        if done:
            self.flush()

        return next_obs, reward, done, info

    def flush(self):
        if len(self._buf_obs) == 0:
            return
        np.savez_compressed(
            self.log_path,
            obs=np.stack(self._buf_obs, axis=0),
            act=np.stack(self._buf_act, axis=0),
            rew=np.asarray(self._buf_rew, dtype=np.float32),
            done=np.asarray(self._buf_done, dtype=np.bool_),
        )
        # clear
        self._buf_obs.clear()
        self._buf_act.clear()
        self._buf_rew.clear()
        self._buf_done.clear()
