import sys
from os.path import dirname
#root_dir = "/home/guillefix/code/inria/RobotLangEnv/"
root_dir = "/gpfswork/rech/imi/usc19dv/captionRLenv/"
sys.path.append(root_dir)

import pickle
import collections
from src.envs.envList import *
from src.envs.descriptions import generate_all_descriptions
from src.envs.env_params import get_env_params
import numpy as np
import pybullet as p
import pickle as pk

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_env_name(task):
    """Construct the env name from parameters."""
    return "LangRobot-v0"

import gym
from gym import spaces
from gym.envs import registration
import numpy as np
from constants import *

import pickle
obs_mod = "obs_cont_single_nocol_noarm_incsize_trim_scaled"
acts_mod = "acts_trim_scaled"
obs_scaler = pickle.load(open(processed_data_folder+obs_mod+"_scaler.pkl", "rb"))
acts_scaler = pickle.load(open(processed_data_folder+acts_mod+"_scaler.pkl", "rb"))
vocab = json.loads(open(processed_data_folder+"npz.annotation.txt.annotation.class_index_reverse.json","r").read())
vocab['72'] = ''


import json
# vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
vocab = json.loads(open(processed_data_folder+"npz.annotation.txt.annotation.class_index_reverse.json","r").read())
vocab['72'] = ''


class LangRobotEnv(ExtendedUR5PlayAbsRPY1Obj):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(LangRobotEnv, self).__init__(obs_scaler=obs_scaler, acts_scaler=acts_scaler, desc_max_len=10, obs_mod="obs_cont_single_nocol_noarm_incsize_trim_scaled")

        # self.action_space = spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)

        self.observation_space = spaces.Dict({
            'obs': spaces.Box(low=-10, high=10, shape=(21,), dtype=np.float32),
            'annotation_emb': spaces.Box(low=-10, high=10, shape=(384*1,), dtype=np.float32),
            'act': spaces.Box(low=-10, high=10, shape=(8,), dtype=np.float32)
        })
        self.annotation_emb = None

    def step(self, action):
        obs, r, done, info = super().step(np.array(action))

        # print(obs[2].shape)
        observation = collections.OrderedDict(
            obs=obs[1][0],
            annotation_emb=self.annotation_emb,
            act=obs[2][0]
        )
        # print(observation)

        return observation, r, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        print("OOOOOOOOOOOOOOOO")
        # print(self.goal_str)
        sent = " ".join([vocab[str(int(x))] for x in self.tokens])
        ann_emb = model.encode(sent)
        self.annotation_emb = ann_emb
        # import pdb; pdb.set_trace()
        observation = collections.OrderedDict(
            obs=self.observation_space["obs"].sample(),
            annotation_emb=self.annotation_emb,
            act=self.observation_space["act"].sample()
        )
        print(observation["annotation_emb"].shape)
        return observation

    def get_metrics(self, num_episodes):
        metrics = []
        success_metric = None
        return metrics, success_metric

if 'LangRobot-v0' in registration.registry.env_specs:
    del registration.registry.env_specs['LangRobot-v0']
registration.register(
    id='LangRobot-v0',
    entry_point=LangRobotEnv,
    max_episode_steps=1000)
