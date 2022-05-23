import os
os.chdir('/home/guillefix/code/')
"""Main binary to train a Behavioral Cloning agent."""
#  pylint: disable=g-long-lambda
import collections
import datetime
import functools
import os

from absl import app
from absl import flags
from absl import logging
import gin
from ibc.environments.block_pushing import block_pushing  # pylint: disable=unused-import
from ibc.environments.block_pushing import block_pushing_discontinuous  # pylint: disable=unused-import
from ibc.environments.particle import particle  # pylint: disable=unused-import
from ibc.ibc import tasks
from ibc.ibc.agents import ibc_policy  # pylint: disable=unused-import
from ibc.ibc.eval import eval_env as eval_env_module
from ibc.ibc.train import get_agent as agent_module
from ibc.ibc.train import get_cloning_network as cloning_network_module
from ibc.ibc.train import get_data as data_module
from ibc.ibc.train import get_eval_actor as eval_actor_module
from ibc.ibc.train import get_learner as learner_module
from ibc.ibc.train import get_normalizers as normalizers_module
from ibc.ibc.train import get_sampling_spec as sampling_spec_module
from ibc.ibc.utils import make_video as video_module
import tensorflow as tf
from tf_agents.system import system_multiprocessing as multiprocessing
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils
from tf_agents.utils import common

VIZIER_KEY = 'success'

strategy = strategy_utils.get_strategy(
  tpu=None, use_gpu=False)
task=["LANG_ROBOT"]
tag=None
add_time=False
viz_img=False
skip_eval=True
learning_rate=1e-4
shared_memory_eval=False
strategy=strategy
checkpoint_interval=100
dataset_path='ibc/data/UR5/tw_data*.tfrecord'
root_dir='./ibc/'
# 'ebm' or 'mse' or 'mdn'.
loss_type='ebm'
# Name of network to train. see get_cloning_network.
network='MLPEBM'
# Training params
batch_size=512
num_iterations=50000
decay_steps=100
replay_capacity=100000
eval_interval=1000
eval_loss_interval=100
eval_episodes=1
fused_train_steps=100
sequence_length=2
uniform_boundary_buffer=0.05
for_rnn=False
flatten_action=True
dataset_eval_fraction=0.0
goal_tolerance=0.02
seed=0
num_envs=1
image_obs=False
# Use this to sweep amount of tfrecords going into training.
# -1 for 'use all'.
max_data_shards=-1
use_warmup=False
#%%
tf.random.set_seed(seed)
if not tf.io.gfile.exists(root_dir):
    tf.io.gfile.makedirs(root_dir)

# Logging.
if tag:
    root_dir = os.path.join(root_dir, tag)
if add_time:
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(root_dir, current_time)

# Define eval env.
eval_envs = []
env_names = []
for task_id in task:
    env_name = eval_env_module.get_env_name(task_id, shared_memory_eval,
                                            image_obs)
    logging.info(('Got env name:', env_name))
    eval_env = eval_env_module.get_eval_env(
        env_name, sequence_length, goal_tolerance, num_envs)
    logging.info(('Got eval_env:', eval_env))
    eval_envs.append(eval_env)
    env_names.append(env_name)

obs_tensor_spec, action_tensor_spec, time_step_tensor_spec = (
  spec_utils.get_tensor_specs(eval_envs[0]))

# Compute normalization info from training data.
create_train_and_eval_fns_unnormalized = data_module.get_data_fns(
  dataset_path,
  sequence_length,
  replay_capacity,
  batch_size,
  for_rnn,
  dataset_eval_fraction,
  flatten_action)
train_data, _ = create_train_and_eval_fns_unnormalized()
(norm_info, norm_train_data_fn) = normalizers_module.get_normalizers(
  train_data, batch_size, env_name, nested_obs=True, nested_actions=False, min_max_actions=True)

import pickle
# pickle.dump(norm_info, open("norm_info_tw_data.pkl","wb"))
# norm_info = pickle.load(open("norm_info_tw_data.pkl","rb"))
#%%

obs_norm_layer = norm_info.obs_norm_layer
act_norm_layer = norm_info.act_norm_layer
import ibc.ibc.utils.constants as constants
from tf_agents.networks import nest_map
def norm_train_data_fn(obs_and_act, nothing):
    obs = obs_and_act[0]
    for img_key in constants.IMG_KEYS:
        if isinstance(obs, dict) and img_key in obs:
            obs[img_key] = tf.image.convert_image_dtype(
                obs[img_key], dtype=tf.float32)
    act = obs_and_act[1]
    normalized_obs = obs_norm_layer(obs)
    if isinstance(obs_norm_layer, nest_map.NestMap):
        normalized_obs, _ = normalized_obs
    normalized_act = act_norm_layer(act)
    if isinstance(act_norm_layer, nest_map.NestMap):
        normalized_act, _ = normalized_act
    return ((normalized_obs, normalized_act), nothing)


# Create normalized training data.
if not strategy:
    strategy = tf.distribute.get_strategy()
per_replica_batch_size = batch_size // strategy.num_replicas_in_sync
create_train_and_eval_fns = data_module.get_data_fns(
  dataset_path,
  sequence_length,
  replay_capacity,
  per_replica_batch_size,
  for_rnn,
  dataset_eval_fraction,
  flatten_action,
  norm_function=norm_train_data_fn,
  max_data_shards=max_data_shards)
# Create properly distributed eval data iterator.
def get_distributed_eval_data(data_fn, strategy):
    """Gets a properly distributed evaluation data iterator."""
    _, eval_data = data_fn()
    dist_eval_data_iter = None
    if eval_data:
        dist_eval_data_iter = iter(
            strategy.distribute_datasets_from_function(lambda: eval_data))
    return dist_eval_data_iter

dist_eval_data_iter = get_distributed_eval_data(create_train_and_eval_fns,
                                              strategy)

#%%

# Create normalization layers for obs and action.
# with strategy.scope():
# Create train step counter.
train_step = train_utils.create_train_step()

# Define action sampling spec.
action_sampling_spec = sampling_spec_module.get_sampling_spec(
    action_tensor_spec,
    min_actions=norm_info.min_actions,
    max_actions=norm_info.max_actions,
    uniform_boundary_buffer=uniform_boundary_buffer,
    act_norm_layer=norm_info.act_norm_layer)

# This is a common opportunity for a bug, having the wrong sampling min/max
# so log this.
logging.info(('Using action_sampling_spec:', action_sampling_spec))

# Define keras cloning network.
cloning_network = cloning_network_module.get_cloning_network(
    network,
    obs_tensor_spec,
    action_tensor_spec,
    norm_info.obs_norm_layer,
    norm_info.act_norm_layer,
    sequence_length,
    norm_info.act_denorm_layer)

# Define tfagent.
agent = agent_module.get_agent(loss_type,
                               time_step_tensor_spec,
                               action_tensor_spec,
                               action_sampling_spec,
                               norm_info.obs_norm_layer,
                               norm_info.act_norm_layer,
                               norm_info.act_denorm_layer,
                               learning_rate,
                               use_warmup,
                               cloning_network,
                               train_step,
                               decay_steps)






###########
#%%

env = eval_envs[0]

from tf_agents.environments import suite_gym

from ibc.environments.lang_robot import lang_robot

task = "LANG_ROBOT"
shared_memory_eval=False
use_image_obs=False
env_name = lang_robot.build_env_name(task)

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
time_step = env.reset()
# time_step = video_env.reset()

from ibc.ibc.utils import strategy_policy
policy = strategy_policy.StrategyPyTFEagerPolicy(agent.policy, strategy=strategy)

# time_step.observation
# action = 0.1*tf.random.normal([2])
# action.numpy()
# while not time_step.is_last():
policy_state = policy.get_initial_state(1)

# policy._policy._wrapped_policy._obs_norm_layer(time_step.observation)

for i in range(10):
    # action = 0.1*tf.random.normal([2])
    # obs_t = obs_from_time_step(time_step)
    # acts = torch.from_numpy(action.numpy()).unsqueeze(0).unsqueeze(0).float().cuda()
    # action = model([obs_t,acts])[0][0][0][0].cpu()
    # action_step = tf_agents.trajectories.PolicyStep(action=action)
    action_step = policy.action(time_step, policy_state)
    # next_time_step = env.step(action)
    # next_time_step = video_env.step(action)
    print(action_step.action)
    next_time_step = env.step(action_step.action)
    # traj = trajectory.from_transition(time_step, action_step, next_time_step)
    # metrics[1](traj)
    time_step = next_time_step
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

# Define bc learner.
bc_learner = learner_module.get_learner(
    loss_type,
    root_dir,
    agent,
    train_step,
    create_train_and_eval_fns,
    fused_train_steps,
    strategy,
    checkpoint_interval=checkpoint_interval)


# bc_learner

# Define eval.
eval_actors, eval_success_metrics = [], []
for eval_env, env_name in zip(eval_envs, env_names):
  env_name_clean = env_name.replace('/', '_')
  eval_actor, success_metric = eval_actor_module.get_eval_actor(
      agent,
      env_name,
      eval_env,
      train_step,
      eval_episodes,
      root_dir,
      viz_img,
      num_envs,
      strategy,
      summary_dir_suffix=env_name_clean)
  eval_actors.append(eval_actor)
  eval_success_metrics.append(success_metric)

get_eval_loss = tf.function(agent.get_eval_loss)

# Get summary writer for aggregated metrics.
aggregated_summary_dir = os.path.join(root_dir, 'eval')
summary_writer = tf.summary.create_file_writer(
    aggregated_summary_dir, flush_millis=10000)
logging.info('Saving operative-gin-config.')
with tf.io.gfile.GFile(
  os.path.join(root_dir, 'operative-gin-config.txt'), 'wb') as f:
f.write(gin.operative_config_str())

# Main train and eval loop.
while train_step.numpy() < num_iterations:
# Run bc_learner for fused_train_steps.
training_step(agent, bc_learner, fused_train_steps, train_step)

if (dist_eval_data_iter is not None and
    train_step.numpy() % eval_loss_interval == 0):
  # Run a validation step.
  validation_step(
      dist_eval_data_iter, bc_learner, train_step, get_eval_loss)

if not skip_eval and train_step.numpy() % eval_interval == 0:

  all_metrics = []
  for eval_env, eval_actor, env_name, success_metric in zip(
      eval_envs, eval_actors, env_names, eval_success_metrics):
    # Run evaluation.
    metrics = evaluation_step(
        eval_episodes,
        eval_env,
        eval_actor,
        name_scope_suffix=f'_{env_name}')
    all_metrics.append(metrics)

    # rendering on some of these envs is broken
    if FLAGS.video and 'kitchen' not in task:
      if 'PARTICLE' in task:
        # A seed with spread-out goals is more clear to visualize.
        eval_env.seed(42)
      # Write one eval video.
      video_module.make_video(
          agent,
          eval_env,
          root_dir,
          step=train_step.numpy(),
          strategy=strategy)

  metric_results = collections.defaultdict(list)
  for env_metrics in all_metrics:
    for metric in env_metrics:
      metric_results[metric.name].append(metric.result())

  with summary_writer.as_default(), \
     common.soft_device_placement(), \
     tf.summary.record_if(lambda: True):
    for key, value in metric_results.items():
      tf.summary.scalar(
          name=os.path.join('AggregatedMetrics/', key),
          data=sum(value) / len(value),
          step=train_step)

summary_writer.flush()


def training_step(agent, bc_learner, fused_train_steps, train_step):
    """Runs bc_learner for fused training steps."""
    reduced_loss_info = None
    if not hasattr(agent, 'ebm_loss_type') or agent.ebm_loss_type != 'cd_kl':
    reduced_loss_info = bc_learner.run(iterations=fused_train_steps)
    else:
    for _ in range(fused_train_steps):
      # I think impossible to do this inside tf.function.
      agent.cloning_network_copy.set_weights(
          agent.cloning_network.get_weights())
      reduced_loss_info = bc_learner.run(iterations=1)

    if reduced_loss_info:
    # Graph the loss to compare losses at the same scale regardless of
    # number of devices used.
    with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
        True):
      tf.summary.scalar(
          'reduced_loss', reduced_loss_info.loss, step=train_step)


def validation_step(dist_eval_data_iter, bc_learner, train_step,
                    get_eval_loss_fn):
    """Runs a validation step."""
    losses_dict = get_eval_loss_fn(next(dist_eval_data_iter))

    with bc_learner.train_summary_writer.as_default(), tf.summary.record_if(
      True):
    common.summarize_scalar_dict(
        losses_dict, step=train_step, name_scope='Eval_Losses/')


def evaluation_step(eval_episodes, eval_env, eval_actor, name_scope_suffix=''):
    """Evaluates the agent in the environment."""
    logging.info('Evaluating policy.')
    with tf.name_scope('eval' + name_scope_suffix):
    # This will eval on seeds:
    # [0, 1, ..., eval_episodes-1]
    for eval_seed in range(eval_episodes):
      eval_env.seed(eval_seed)
      eval_actor.reset()  # With the new seed, the env actually needs reset.
      eval_actor.run()

    eval_actor.log_metrics()
    eval_actor.write_metric_summaries()
    return eval_actor.metrics
