from extra_utils import distribute_tasks
import os, sys
from constants import *
from ibc.evaluate_model import run
import argparse
print("EEEEEEEEEEE")
sys.stdout.flush()

parser = argparse.ArgumentParser(description='parallel eval')
parser.add_argument('--base_filenames_file', type=str, default='base_filenames_single_objs_filtered.txt', help='file listing demo sequence ids')
parser.add_argument('--sample_goals', action='store_true')
parser.add_argument('--num_tasks', type=int, default=1, help='number of tasks (overriden by number of sequence ids if base_filenames_file is not None)')
parser.add_argument('--num_repeats', type=int, default=1, help='number of times each demo should be used')
parser.add_argument('--goal_str', type=str, default=None)
parser.add_argument('--savepath', type=str, default=".")
parser.add_argument('--save_eval_results', action='store_true')
parser.add_argument('--restore_objects', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--session_id', type=str)
parser.add_argument('--rec_id', type=str)
parser.add_argument('--varying_args', type=str, default='session_id,rec_id')
parser.add_argument('--max_episode_length', type=int, default=3000)
print("OOOOOOOOOOo")


#######################
######## setup ########
#######################

args = parser.parse_args()
print("UUUUUUUUUU")
sys.stdout.flush()

## distributing tasks accross nodes ##
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
print(rank)

#TODO: add task where we sample a random instruction

tasks = []

common_args = vars(args).copy()
del common_args["base_filenames_file"]
del common_args["num_repeats"]
del common_args["sample_goals"]
del common_args["num_tasks"]
if args.base_filenames_file is not None:
    with open(processed_data_folder+args.base_filenames_file, "r") as f:
        filenames = [x[:-1] for x in f.readlines()] # to remove new lines
    num_tasks = len(filenames)
    tasks = args.num_repeats*list(map(lambda x: {**common_args, "session_id": x.split("_")[1], "rec_id": x.split("_")[5]}, filenames))
elif args.sample_goals:
    from extra_utils.run_utils import generate_goal
    tasks = args.num_repeats*[{**common_args, "goal_str": generate_goal()} for i in range(args.num_tasks)]

    #filenames = filenames[:2]

import tensorflow as tf

#common_args = {"restore_objects": True}
tasks = distribute_tasks(tasks, rank, size)
print(tasks)
class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

for task in tasks:
    args = Struct(**task)
    policy = tf.compat.v2.saved_model.load('/home/guillefix/code/awo_testing3_old')
    run(policy,args)
print("Finished. Rank: "+str(rank))
