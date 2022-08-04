import pybullet as p
import numpy as np
import os, sys
from constants import *
import tensorflow as tf
from tf_agents.environments import suite_gym
from pathlib import Path

from ibc.environments.lang_robot import lang_robot
print("AWOO")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

from ibc.ibc.agents.ibc_policy import MappedCategorical
print("OWAA")
sys.stdout.flush()


def run(policy, args):
    os.chdir('/home/guillefix/code/')
    # human_data = np.load("/home/guillefix/code/inria/UR5/Guillermo/obs_act_etc/8/data.npz", allow_pickle=True)
    # session_id = "Guillermo1"
    # rec_id = "626"
    human_data = np.load("/home/guillefix/code/inria/UR5/"+args.session_id+"/obs_act_etc/"+args.rec_id+"/data.npz", allow_pickle=True)
    traj_data_obss = np.load(processed_data_folder+"UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data.obs_cont_single_nocol_noarm_incsize_trim_scaled.npy")
    traj_data_actss = np.load(processed_data_folder+"UR5_"+args.session_id+"_obs_act_etc_"+args.rec_id+"_data.acts_trim_scaled.npy")
    # from sentence_transformers import SentenceTransformer
    import json
    # vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
    vocab = json.loads(open(processed_data_folder+"npz.annotation.txt.annotation.class_index_reverse.json","r").read())
    # vocab['66'] = ''
    # root_folder = "/home/guillefix/code/inria/UR5_processed/"
    # filename="UR5_Guillermo_obs_act_etc_8_data"
    # disc_cond = np.load(root_folder+filename+".disc_cond.npy")
    # sents = [vocab[str(int(x))] for x in disc_cond]
    # sent1 = " ".join(filter(lambda x: x!='', sents[:11]))
    # sent2 = " ".join(filter(lambda x: x!='', sents[11:]))
    # with open(root_folder+filename+".annotation.txt","r") as f:
    #     annotation = f.read()
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # sentence_embeddings = model.encode([sent1,sent2])
    # sentence_embedding1, sentence_embedding2 = sentence_embeddings[0], sentence_embeddings[1]
    # ann_emb1 = sentence_embedding1
    # ann_emb2 = sentence_embedding2
    # annotation_emb = np.concatenate([ann_emb1, ann_emb2])
    # np.save("annotation_emb.npy", annotation_emb)
    # annotation_emb = np.load("annotation_emb.npy")

    # policy = tf.compat.v2.saved_model.load('/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220215-150051/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc/policies/20220218-195350/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc/policies/20220223-033003/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_policies/20220308-002340/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220524-013833/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220602-063734/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220603-060321/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220603-230841/policies/policy/')
    #policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220606-232320/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220609-001705/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220630-143918/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/tmp/ibc_logs/mlp_ebm/ibc_dfo/20220630-145820/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220701-170540/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220704-221712/policies/policy/')
    # import pdb; pdb.set_trace()
    # collect_policy = ibc_policy.IbcPolicy(
    #     time_step_spec=time_step_spec,
    #     action_spec=action_spec,
    #     action_sampling_spec=action_sampling_spec,
    #     actor_network=cloning_network,
    #     late_fusion=late_fusion,
    #     obs_norm_layer=self._obs_norm_layer,
    #     act_denorm_layer=self._act_denorm_layer,
    # )


    # task = "LANG_ROBOT"
    task = "LANG_ROBOT_LANG"
    shared_memory_eval=False
    use_image_obs=False
    # sequence_length=3
    sequence_length=1
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
    #policy_state = policy.get_initial_state(1)
    policy_state = policy.get_initial_state()

    # policy._policy._wrapped_policy._obs_norm_layer(time_step.observation)


    import tf_agents

    # n=2
    n=sequence_length

    # env.action_space.high.shape
    # env._env._env._gym_env.env.env.action_space.low.shape
    # env._env._env._gym_env.env.annotation_emb = annotation_emb
    # env._env._env._gym_env.env.ex_data = human_data
    if args.render:
        env._env._env._gym_env.env.render()
    time_step = env.reset()
    # import pdb; pdb.set_trace()
    time_step.observation['obs'] = traj_data_obss[0:n].astype(np.float32)
    time_step.observation['act'] = traj_data_actss[0:n].astype(np.float32)
    # time_step.observation['annotation_emb'] = time_step.observation['annotation_emb'][:1]
    # time_step.observation['annotation_emb'] = time_step.observation['annotation_emb'][:n]
    print(time_step.observation['obs'])
    print(time_step.observation['act'])
    # import pdb; pdb.set_trace()
    #env._env._env._gym_env.env.reset(description="Paint green dog red")
    goal_str = args.goal_str
    if args.goal_str is None:
        goal_str = human_data["goal_str"][0]
    env._env._env._gym_env.env.reset(description=goal_str, o=human_data["obs"][0], info_reset=None, joint_poses=human_data["joint_poses"][0], objects=human_data['obj_stuff'][0], restore_objs=args.restore_objects)
    # time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
    #                                           discount=tf.expand_dims(time_step.discount,0),
    #                                           observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation_emb':tf.expand_dims(time_step.observation['annotation_emb'],0), 'act':tf.expand_dims(time_step.observation['act'],0)})
    time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
                                              discount=tf.expand_dims(time_step.discount,0),
                                              observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation':tf.expand_dims(time_step.observation['annotation'],0), 'act':tf.expand_dims(time_step.observation['act'],0)})

    traj_data_actss = traj_data_actss[2:]
    traj_data_obss = traj_data_obss[2:]

    # env.render()
    # env._env._env._gym_env.env.observation_space[1]
    #%%
    # env._env._env._gym_env.env.action_space

    # policy._optimize_again = True
    # policy._use_dfo = False
    # policy._use_langevin = True
    # policy_step = policy.distribution(time_step, policy_state)
    # mapped_values = policy_step.action._mapped_values
    # action1 = np.random.randn(8)
    # action2 = np.random.randn(8)
    # awo=[action1, action2]
    # import pdb; pdb.set_trace()

    achieved_goal_end = False
    for t in range(args.max_episode_length):
        print(t)
        # action = 0.1*tf.random.normal([2])
        # obs_t = obs_from_time_step(time_step)
        # acts = torch.from_numpy(action.numpy()).unsqueeze(0).unsqueeze(0).float().cuda()
        # action = model([obs_t,acts])[0][0][0][0].cpu()
        # action_step = tf_agents.trajectories.PolicyStep(action=action)
        # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220704-035503/policies/policy/')
        # policy_state = policy.get_initial_state(1)
        # action = mapped_values[np.random.randint(len(mapped_values))]
        # import pdb; pdb.set_trace()
        # action_step = policy.action(time_step, policy_state)
        policy_step = policy.distribution(time_step, policy_state)
        # action = policy_step.action.sample(1)[0]
        # action = policy_step.action.mode()[0]
        action = policy_step.action.sample()[0]
        action = tf.expand_dims(action,0)
        # print(action)
        # params = policy_step.action.logits_parameter()
        # print(params.numpy().min())
        # print(params.numpy().max())
        # print(policy_state)
        # print(policy.train_step.numpy())
        # next_time_step = env.step(action)
        # next_time_step = video_env.step(action)
        # action = np.array(action_step.action.numpy().tolist())
        # action = awo[np.random.randint(2)]
        # action = awo[(i//20)%2]
        # action = np.expand_dims(action,0)
        # action = np.expand_dims(action.numpy(),0)
        # print(action)
        # action = np.array(action.tolist())
        # action = np.array(action.numpy().tolist())
        # print(action.shape)
        # action = traj_data_actss[i:i+1]
        # print(action.shape)
        # print(action)
        next_time_step = env.step(action)
        # traj = trajectory.from_transition(time_step, action_step, next_time_step)
        # metrics[1](traj)
        # policy_state = action_step.state
        time_step = next_time_step
        # time_step.observation['obs'] = traj_data_obss[t:t+n].astype(np.float32)
        # time_step.observation['act'] = traj_data_actss[t:t+n].astype(np.float32)
        # # time_step.observation['obs'] = traj_data_obss[i%(100-n):i%(100-n)+n].astype(np.float32)
        # # time_step.observation['act'] = traj_data_actss[i%(100-n):i%(100-n)+n].astype(np.float32)
        # if (i//20) % 2 == 0: k=0
        # else: k=100
        # k=t
        # time_step.observation['obs'] = traj_data_obss[k:k+n].astype(np.float32)
        # time_step.observation['act'] = traj_data_actss[k:k+n].astype(np.float32)
        # # time_step.observation['obs'] = traj_data_obss[0:0+n].astype(np.float32)
        # # time_step.observation['act'] = traj_data_actss[0:0+n].astype(np.float32)
        # print(time_step.observation['annotation_emb'].shape)
        # time_step.observation['obs'] = time_step.observation['obs'][0,:1]
        # time_step.observation['act'] = time_step.observation['act'][0,:1]
        # time_step.observation['annotation_emb'] = time_step.observation['annotation_emb'][0,:n]
        # time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
        #                                           discount=tf.expand_dims(time_step.discount,0),
        #                                           observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation_emb':tf.expand_dims(time_step.observation['annotation_emb'],0), 'act':tf.expand_dims(time_step.observation['act'],0)})
        # time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
        #                                           discount=tf.expand_dims(time_step.discount,0),
        #                                           observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation_emb':tf.expand_dims(time_step.observation['annotation_emb'],0), 'act':tf.expand_dims(time_step.observation['act'],0)})
        time_step = tf_agents.trajectories.time_step.TimeStep(step_type=tf.expand_dims(time_step.step_type,0), reward=tf.expand_dims(time_step.reward,0),
                                                  discount=tf.expand_dims(time_step.discount,0),
                                                  observation={'obs':tf.expand_dims(time_step.observation['obs'],0), 'annotation':tf.expand_dims(time_step.observation['annotation'],0), 'act':tf.expand_dims(time_step.observation['act'],0)})
        print(time_step.observation['obs'])
        print(time_step.observation['act'])
        done = env._env._env._gym_env.env.done
        reward = env._env._env._gym_env.env.reward
        success = reward > 0
        print(goal_str+": ",success)
        achieved_goal_end = success
        if done:
            break
        # print(time_step)
    if args.save_eval_results:
        varying_args = args.varying_args.split(",")
        print(Path(args.savepath+"/results"))
        if not Path(args.savepath+"/results").is_dir():
            os.mkdir(args.savepath+"/results")
        filename = args.savepath+"/results/"+"eval_"
        args_dict = vars(args)
        for k in varying_args:
            filename += str(args_dict[k])+"_"
        filename += "_".join(goal_str.split(" "))+".txt"
        if os.path.exists(filename):
            with open(filename, "a") as f:
                f.write(str(achieved_goal_end)+","+str(t)+"\n")
        else:
            with open(filename, "w") as f:
                f.write("achieved_goal_end,num_steps"+"\n")
                f.write(str(achieved_goal_end)+","+str(t)+"\n")

    # video_env.close()

    # metrics[1]._env.succeeded
    # metrics[0][0].result()
    # metrics[1].result()
    # metrics[1]._np_state.success
    # metrics[1]._buffer



    # agent.policy
    #%%
    ######################


if __name__ == "__main__":
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/ibc_logs/mlp_ebm/ibc_dfo/20220704-221712/policies/policy/')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testing3')
    #policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_longer_seq_len')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_smallerlr')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_bigbs_lr2_decayfast2')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_bigbs_lr2_decayfas87')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_bigbs_lr2_decayfast7')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_bigbs_lr2_decayfas12')
    policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_lang')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testin_bigbs_lr2_decayfast16')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testing3_old')
    # policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testing4')
    # from tf_agents.policies import greedy_policy
    # policy = greedy_policy.GreedyPolicy(policy)

    args_dir = {"session_id": "Guillermo1", "rec_id": "626", "render": True, "goal_str": None,
                "save_eval_results": False, "max_episode_length": 3000, "restore_objects": True}
    # args_dir = {"session_id": "Guillermo1", "rec_id": "629", "render": True, "goal_str": None,
    #             "save_eval_results": False, "max_episode_length": 3000, "restore_objects": True}

    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)
    args = Struct(**args_dir)
    run(policy, args)
