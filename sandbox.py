import os
os.chdir('/home/guillefix/code/')
# import torch
# model = torch.jit.load('ibc/compiled_jit.pth')
import tensorflow as tf
import glob
import numpy as np
from tf_agents.trajectories import trajectory
import tf_agents
# tf_agents.trajectories.PolicyStep
#%%
# filenames=glob.glob("ibc/data/block_push_states_location/*tfrecord")
filenames=glob.glob("ibc/data/UR5/*tfrecord")
raw_dataset = tf.data.TFRecordDataset(filenames)
# for raw_record in raw_dataset.take(1):
obss = []
actss = []
for i,raw_record in enumerate(raw_dataset):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    # print(example.features.feature.keys())
    # print(example.features.feature["observation/block_orientation"].float_list.value)
    # print(example.features.feature["observation/block_translation"].float_list.value[1])
    # print(example.features.feature["observation/effector_translation"])
    # print(example.features.feature["observation/target_translation"])
    # print(example.features.feature["observation/effector_target_translation"])
    # print(example.features.feature["observation/target_orientation"])
    # print(example.features.feature["action"])
    obs = []
    for obs_feature in ["block_orientation", "block_translation", "effector_translation", "target_translation", "effector_target_translation", "target_orientation"]:
       obs += example.features.feature["observation/"+obs_feature].float_list.value

    obss.append(obs)

    acts = []
    for acts_feature in ["action"]:
       acts += example.features.feature[acts_feature].float_list.value
    actss.append(acts)


obss = np.array(obss)
actss = np.array(actss)

obss.shape
actss.shape

np.save("ibc/data/block_push_states_location/obs", obss)
np.save("ibc/data/block_push_states_location/acts", actss)

#%%

import os
os.chdir('/home/guillefix/code/')
from tf_agents.environments import suite_gym

from ibc.environments.block_pushing import block_pushing

task = "PUSH"
shared_memory_eval=False
use_image_obs=False
env_name = block_pushing.build_env_name(
    task, shared_memory_eval, use_image_obs=use_image_obs)

env = suite_gym.load(env_name)

env.action_spec()

from ibc.ibc.utils import oss_mp4_video_wrapper as mp4_video_wrapper
root_dir = "./ibc/"
video_path = os.path.join(root_dir, 'videos', 'ttl=7d', 'vid_0.mp4')
if not hasattr(env, 'control_frequency'):
    # Use this control freq for d4rl envs, which don't have a control_frequency
    # attr.
    control_frequency = 30
else:
    control_frequency = env.control_frequency
video_env = mp4_video_wrapper.Mp4VideoWrapper(
  env, control_frequency, frame_interval=1, video_filepath=video_path)


import tensorflow as tf

tf.random.normal([2])

metrics = env.get_metrics(1)

from ibc.environments.block_pushing.oracles.oriented_push_oracle import OrientedPushOracle

policy = OrientedPushOracle(env)
#%%
# policy = OrientedPushOracle(env)

import pickle
# root_folder2="/home/guillefix/code/ibc/"
# acts_scaler = pickle.load(open(root_folder2+"UR5_processed/acts_scaled_scaler.pkl", "rb"))
# obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_scaled_scaler.pkl", "rb"))

def obs_from_time_step(time_step):
    obs = []
    for obs_feature in ["block_orientation", "block_translation", "effector_translation", "target_translation", "effector_target_translation", "target_orientation"]:
        obs += time_step.observation[obs_feature].tolist()
    obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).unsqueeze(0).float().cuda()
    return obs_t

obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).unsqueeze(0).float().cuda()
acts = torch.randn(2).unsqueeze(0).unsqueeze(0).float().cuda()

obs_t.shape
acts.shape

model([obs_t,acts])[0]
model([obs_t,acts])[0]

# reset() creates the initial time_step and resets the environment.
# time_step = env.reset()
time_step = video_env.reset()

time_step.observation
action = 0.1*tf.random.normal([2])
# action.numpy()
while not time_step.is_last():
    # action = 0.1*tf.random.normal([2])
    obs_t = obs_from_time_step(time_step)
    acts = torch.from_numpy(action.numpy()).unsqueeze(0).unsqueeze(0).float().cuda()
    action = model([obs_t,acts])[0][0][0][0].cpu()
    action_step = tf_agents.trajectories.PolicyStep(action=action)
    # action_step = policy.action(time_step)
    # next_time_step = env.step(action)
    next_time_step = video_env.step(action)
    # next_time_step = env.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)
    metrics[1](traj)
    time_step = next_time_step
    # print(time_step)

video_env.close()

metrics[1]._env.succeeded
metrics[0][0].result()
metrics[1].result()
metrics[1]._np_state.success
metrics[1]._buffer

# from ibc.ibc.utils import make_video as video_module
