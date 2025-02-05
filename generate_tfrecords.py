import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
from tf_agents.utils import example_encoding


# tf.__version__

# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(values):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(values):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(values):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

# def _int32_feature(values):
#     """Returns an int32_list from a bool / enum / int / uint."""
#     return tf.train.Feature(int32_list=tf.train.Int32List(value=values))
from tf_agents.trajectories.trajectory import Trajectory
output_spec=Trajectory(
    discount=tf.TensorSpec(shape=(1,), dtype=tf.float32),
    step_type=tf.TensorSpec(shape=(1,), dtype=tf.int64),
    next_step_type=tf.TensorSpec(shape=(1,), dtype=tf.int64),
    # observation={"obs":tf.TensorSpec(shape=(125,), dtype=tf.float32),
    #              "annotation_emb":tf.TensorSpec(shape=(384,), dtype=tf.float32)},
    observation={"obs":tf.TensorSpec(shape=(71,), dtype=tf.float32),
                 "annotation_emb":tf.TensorSpec(shape=(384*2,), dtype=tf.float32)},
    action=tf.TensorSpec(shape=(8,), dtype=tf.float32),
    reward=tf.TensorSpec(shape=(1,), dtype=tf.float32),
    policy_info={}
    )

# encoder = example_encoding.get_example_encoder(output_spec)
# serializer = example_encoding.get_example_serializer(output_spec)

def serialize_example(obs,acts,ann_emb):
    features = {
        "discount": _float_feature([1.0]),
        "step_type": _int64_feature([1]),
        "next_step_type": _int64_feature([1]),
        "observation/obs": _float_feature(obs),
        "observation/annotation_emb": _float_feature(ann_emb),
        "action": _float_feature(acts),
        "reward": _float_feature([1.0])
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    return example_proto.SerializeToString()
    # return example_proto.features.feature
    # return example_proto


#%%

#####
import numpy as np
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

root_folder = "/home/guillefix/code/inria/UR5_processed/"
filenames=[x[:-1] for x in open("/home/guillefix/code/inria/UR5_processed/base_filenames.txt","r").readlines()]

filename = filenames[0]
import json
# vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index.json","r").read())
vocab = json.loads(open("/home/guillefix/code/inria/UR5_processed/acts.npy.annotation.class_index_reverse.json","r").read())
vocab['66'] = ''

def generator():
    for filename in filenames:
        obss = np.load(root_folder+filename+".obs_cont.npy")
        actss = np.load(root_folder+filename+".npz.acts.npy")
        disc_cond = np.load(root_folder+filename+".disc_cond.npy")
        sents = [vocab[str(int(x))] for x in disc_cond]
        sent1 = " ".join(filter(lambda x: x!='', sents[:11]))
        sent2 = " ".join(filter(lambda x: x!='', sents[11:]))
        # with open(root_folder+filename+".annotation.txt","r") as f:
        #     annotation = f.read()
        sentence_embeddings = model.encode([sent1,sent2])
        sentence_embedding1, sentence_embedding2 = sentence_embeddings[0], sentence_embeddings[1]
        # sentence_embedding2 = model.encode(sent2)
        for i,obs in enumerate(obss):
            acts = actss[i]
            ann_emb1 = sentence_embedding1
            ann_emb2 = sentence_embedding2
            ann_emb = np.concatenate([ann_emb1, ann_emb2])
            # thing = serialize_example(obs,acts,ann_emb)
            # yield {k:v.float_list.value if v.float_list.value != [] else v.int64_list.value for k,v in thing.items()}
            yield serialize_example(obs,acts,ann_emb)


thing = next(generator())
# thing['step_type'].float_list.value
# {k:v.float_list.value if v.float_list.value != [] else v.int64_list.value for k,v in thing.items()}
# tf.type_spec_from_value(thing.features.feature)
# len(thing.features.feature["observation/obs"].float_list.value)
# len(thing.features.feature["observation/annotation_emb"].float_list.value)
# type(thing)
# output_signature=(
#     tf.TensorSpec(shape=(1,), dtype=tf.float32),
#     tf.TensorSpec(shape=(1,), dtype=tf.int64),
#     tf.TensorSpec(shape=(1,), dtype=tf.int64),
#     # tf.TensorSpec(shape=(125,), dtype=tf.float32),
#     tf.TensorSpec(shape=(71,), dtype=tf.float32),
#     # tf.TensorSpec(shape=(384,), dtype=tf.float32),
#     tf.TensorSpec(shape=(384*2,), dtype=tf.float32),
#     tf.TensorSpec(shape=(8,), dtype=tf.float32),
#     tf.TensorSpec(shape=(1,), dtype=tf.float32),
#     )
output_signature={
    "discount": tf.TensorSpec(shape=(1,), dtype=tf.float32),
    "step_type": tf.TensorSpec(shape=(1,), dtype=tf.int64),
    "next_step_type": tf.TensorSpec(shape=(1,), dtype=tf.int64),
    # tf.TensorSpec(shape=(125,), dtype=tf.float32),
    "observation/obs": tf.TensorSpec(shape=(71,), dtype=tf.float32),
    # tf.TensorSpec(shape=(384,), dtype=tf.float32),
    "observation/annotation_emb": tf.TensorSpec(shape=(384*2,), dtype=tf.float32),
    "action": tf.TensorSpec(shape=(8,), dtype=tf.float32),
    "reward": tf.TensorSpec(shape=(1,), dtype=tf.float32),
    }
serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())
    # generator, output_signature=output_signature)
    # generator, output_signature=output_spec)

# from tensorflow.python.data.ops import dataset_ops
# dataset_ops.get_structure(dataset_shard)
# dataset_shard.element_spec
# list(serialized_features_dataset.take(1))
# next(iter(serialized_features_dataset))



#%%

# tf.data.experimental.save(serialized_features_dataset, 'data/UR5')

# for record in serialized_features_dataset.take(10):
#     print(record)

#%%
from tf_agents.utils import example_encoding_dataset
import pickle
num_shards = 10
for i in range(0,1):
    dataset_shard = serialized_features_dataset.shard(num_shards=num_shards, index=i)
    filename = 'data/UR5/tw_data_'+str(i)+'.tfrecord'
    spec_filename = filename + ".spec"
    # example_encoding_dataset.encode_spec_to_file(spec_filename, dataset_shard.element_spec)
    example_encoding_dataset.encode_spec_to_file(spec_filename, output_spec)
    # print(dataset_shard.element_spec)
    # writer = tf.data.experimental.TFRecordWriter(filename)
    with tf.io.TFRecordWriter(filename) as writer:
        for record in dataset_shard:
            writer.write(record.numpy())


1
data_iter = iter(dataset)
next(data_iter)

SimpleSpec = collections.namedtuple("SimpleSpec", ("step_type", "value"))

#%%
### Process some existing dataset

import glob
# dataset_name = "block_push_states_location"
dataset_name = "UR5"
path_to_shards=glob.glob("data/"+dataset_name+"/*tfrecord")
dataset = tf.data.Dataset.from_tensor_slices(path_to_shards).repeat()


buffer_size_per_shard = 100
seq_len = 1
def interleave_func(shard):
    dataset = tf.data.TFRecordDataset(
        shard, buffer_size=buffer_size_per_shard).cache().repeat()
    dataset = dataset.window(seq_len, shift=1, stride=1, drop_remainder=True)
    return dataset.flat_map(
        lambda window: window.batch(seq_len, drop_remainder=True))
dataset = dataset.interleave(interleave_func,
                           deterministic=True,
                           cycle_length=len(path_to_shards),
                           block_length=1,
                           num_parallel_calls=1)

from tf_agents.utils import example_encoding_dataset
from tf_agents.utils import example_encoding
specs = []
for dataset_file in path_to_shards:
    spec_path = dataset_file + example_encoding_dataset._SPEC_FILE_EXTENSION
    dataset_spec = example_encoding_dataset.parse_encoded_spec_from_file(
        spec_path)
    specs.append(dataset_spec)
    if not all([dataset_spec == spec for spec in specs]):
        raise ValueError('One or more of the encoding specs do not match.')

decoder = example_encoding.get_example_decoder(specs[0], batched=True,
                                             compress_image=True)

decoder
dataset = dataset.map(decoder, num_parallel_calls=1)

data_iter = iter(dataset)

traj = next(data_iter)
traj
traj.observation
tf.nest.flatten(traj.observation)
