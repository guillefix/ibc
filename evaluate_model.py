import pybullet as p
import numpy as np
import os
os.chdir('/home/guillefix/code/')
human_data = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/11/data.npz", allow_pickle=True)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
annotation = human_data['goal_str'][0]
annotation = "Put white banana on the shelf"
annotation_emb = model.encode(annotation)
np.save("annotation_emb.npy", annotation_emb)
# annotation_emb = np.load("annotation_emb.npy")

from tf_agents.environments import suite_gym

from ibc.environments.lang_robot import lang_robot

import tensorflow as tf

from ibc.ibc.agents.ibc_policy import MappedCategorical

# policy = tf.compat.v2.saved_model.load('/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220215-150051/policies/policy/')
# policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc/policies/20220218-195350/policy/')
policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc/policies/20220223-033003/policies/policy/')

task = "LANG_ROBOT"
shared_memory_eval=False
use_image_obs=False
sequence_length=2
goal_tolerance=0.02
num_envs=1

from ibc.ibc.eval import eval_env as eval_env_module
env_name = eval_env_module.get_env_name(task, shared_memory_eval, use_image_obs)
env = eval_env_module.get_eval_env(
    env_name, sequence_length, goal_tolerance, num_envs)

# env.action_spec()

# from ibc.ibc.utils import oss_mp4_video_wrapper as mp4_video_wrapper
# root_dir = "./ibc/"
# video_path = os.path.join(root_dir, 'videos', 'ttl=7d', 'vid_0.mp4')
# if not hasattr(env, 'control_frequency'):
#     # Use this control freq for d4rl envs, which don't have a control_frequency
#     # attr.
#     control_frequency = 30
# else:
#     control_frequency = env.control_frequency
# video_env = mp4_video_wrapper.Mp4VideoWrapper(
#   env, control_frequency, frame_interval=1, video_filepath=video_path)
#

metrics = env.get_metrics(1)

# from ibc.environments.block_pushing.oracles.oriented_push_oracle import OrientedPushOracle

# policy = OrientedPushOracle(env)
# policy = OrientedPushOracle(env)

# import pickle
# # root_folder2="/home/guillefix/code/ibc/"
# # acts_scaler = pickle.load(open(root_folder2+"UR5_processed/acts_scaled_scaler.pkl", "rb"))
# # obs_scaler = pickle.load(open(root_folder2+"UR5_processed/obs_scaled_scaler.pkl", "rb"))
#
# def obs_from_time_step(time_step):
#     obs = []
#     for obs_feature in ["block_orientation", "block_translation", "effector_translation", "target_translation", "effector_target_translation", "target_orientation"]:
#         obs += time_step.observation[obs_feature].tolist()
#     obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).unsqueeze(0).float().cuda()
#     return obs_t
#
# obs_t = torch.from_numpy(np.array(obs)).unsqueeze(0).unsqueeze(0).float().cuda()
# acts = torch.randn(2).unsqueeze(0).unsqueeze(0).float().cuda()
#
# obs_t.shape
# acts.shape
#
# model([obs_t,acts])[0]
# model([obs_t,acts])[0]

# reset() creates the initial time_step and resets the environment.
# time_step = video_env.reset()

# from ibc.ibc.utils import strategy_policy
# policy = strategy_policy.StrategyPyTFEagerPolicy(agent.policy, strategy=strategy)

# time_step.observation
# action = 0.1*tf.random.normal([2])
# action.numpy()
# while not time_step.is_last():
policy_state = policy.get_initial_state(1)

# policy._policy._wrapped_policy._obs_norm_layer(time_step.observation)


import tf_agents

# env.action_space.high.shape
# env._env._env._gym_env.env.env.action_space.low.shape
env._env._env._gym_env.env.annotation_emb = annotation_emb
env._env._env._gym_env.env.ex_data = human_data
time_step = env.reset()
time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
                                          discount=tf.expand_dims(time_step.discount,0),
                                          observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation_emb':tf.expand_dims(time_step.observation['annotation_emb'],0)})

#%%
# env._env._env._gym_env.env.action_space

for i in range(1000):
    print(i)
    # action = 0.1*tf.random.normal([2])
    # obs_t = obs_from_time_step(time_step)
    # acts = torch.from_numpy(action.numpy()).unsqueeze(0).unsqueeze(0).float().cuda()
    # action = model([obs_t,acts])[0][0][0][0].cpu()
    # action_step = tf_agents.trajectories.PolicyStep(action=action)
    action_step = policy.action(time_step, policy_state)
    # next_time_step = env.step(action)
    # next_time_step = video_env.step(action)
    acts = action_step.action.numpy().tolist()[0]
    action = [acts[0],acts[1],acts[2]] + list(p.getEulerFromQuaternion(acts[3:7])) + [acts[7]]
    print(action)
    next_time_step = env.step(action)
    # traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # metrics[1](traj)
    policy_state = action_step.state
    time_step = next_time_step
    time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
                                              discount=tf.expand_dims(time_step.discount,0),
                                              observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation_emb':tf.expand_dims(time_step.observation['annotation_emb'],0)})
    # print(time_step)

video_env.close()

metrics[1]._env.succeeded
metrics[0][0].result()
metrics[1].result()
metrics[1]._np_state.success
metrics[1]._buffer



agent.policy
#%%
######################
