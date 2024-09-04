# Copyright 2020 The FedLearner Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
# pylint: disable=no-else-return, inconsistent-return-statements

import tensorflow.compat.v1 as tf
import fedlearner.trainer as flt
from fedlearner.privacy.mpcnn.nn_mpc_layer import NNMPCLayer

ROLE = 'leader'

parser = flt.trainer_worker.create_argument_parser()
parser.add_argument('--batch-size', type=int, default=32,
                    help='Training batch size.')
args = parser.parse_args()


def input_fn(bridge, trainer_master=None):
    dataset = flt.data.DataBlockLoader(
        args.batch_size, ROLE, bridge, trainer_master).make_dataset()

    def parse_fn(example):
        feature_map = {"x_{0}".format(i): tf.VarLenFeature(
            tf.int64) for i in range(512)}
        feature_map["example_id"] = tf.FixedLenFeature([], tf.string)
        feature_map["y"] = tf.FixedLenFeature([], tf.int64)
        features = tf.parse_example(example, features=feature_map)
        return features, dict(y=features.pop('y'))

    dataset = dataset.map(map_func=parse_fn,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return dataset


def serving_input_receiver_fn():
    feature_map = {"x_{0}".format(i): tf.VarLenFeature(
        tf.int64) for i in range(512)}
    feature_map["example_id"] = tf.FixedLenFeature([], tf.string)

    record_batch = tf.placeholder(dtype=tf.string, name='examples')
    features = tf.parse_example(record_batch, features=feature_map)
    features['act1_f'] = tf.placeholder(dtype=tf.float32, name='act1_f')
    receiver_tensors = {
        'examples': record_batch,
        'act1_f': features['act1_f']
    }
    return tf.estimator.export.ServingInputReceiver(
        features, receiver_tensors)


def model_fn(model, features, labels, mode):
    """Model Builder of wide&deep learning models
    Args:
    Returns
    """
    global_step = tf.train.get_or_create_global_step()

    num_slot = 64
    x = dict()
    for i in range(num_slot):
        x_name = "x_{}".format(i)
        x[x_name] = features[x_name]

    fid_size, embed_size = 101, 16
    embeddings = [
        tf.get_variable(
            'slot_emb{0}'.format(i), shape=[fid_size, embed_size],
            dtype=tf.float32,
            initializer=tf.random_uniform_initializer(-0.01, 0.01))
        for i in range(num_slot)]
    embed_output = tf.concat(
        [
            tf.nn.embedding_lookup_sparse(
                embeddings[i], x['x_{}'.format(i)], sp_weights=None,
                combiner='mean')
            for i in range(num_slot)],
        axis=1)

    lr = 0.1
    mpc_output_dim = 512
    input_size = num_slot * embed_size
    nn_mpc_layer = NNMPCLayer(args.batch_size, output_dim=mpc_output_dim, other_party_input_dim=input_size, model=model, mode=mode)
    result = nn_mpc_layer(embed_output)

    fc2_size = 64
    b1l = tf.get_variable(
        'b1l', shape=[mpc_output_dim], dtype=tf.float32,
        initializer=tf.zeros_initializer())
    w2 = tf.get_variable(
        'w2', shape=[mpc_output_dim, fc2_size], dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))
    b2 = tf.get_variable(
        'b2', shape=[fc2_size], dtype=tf.float32,
        initializer=tf.zeros_initializer())
    w3 = tf.get_variable(
        'w3', shape=[fc2_size, 2], dtype=tf.float32,
        initializer=tf.random_uniform_initializer(-0.01, 0.01))

    act1_l = tf.nn.relu(tf.nn.bias_add(result, b1l))
    act2_l = tf.nn.bias_add(tf.matmul(act1_l, w2), b2)
    logits = tf.matmul(act2_l, w3)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return model.make_spec(mode, predictions=logits)

    y = labels['y']
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=y, logits=logits)
    loss = tf.math.reduce_mean(loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        auc_pair = tf.metrics.auc(y, logits[:, 1])
        return model.make_spec(
            mode, loss=loss, eval_metric_ops={'auc': auc_pair})

    # mode == tf.estimator.ModeKeys.TRAIN:
    logging_hook = tf.train.LoggingTensorHook(
        {"loss": loss}, every_n_iter=10)
    metric_hook = flt.GlobalStepMetricTensorHook(tensor_dict={"loss": loss},
                                                 every_steps=10)

    optimizer = tf.train.GradientDescentOptimizer(lr)
    train_op = model.minimize(optimizer, loss, global_step=global_step)
    return model.make_spec(mode, loss=loss, train_op=train_op,
                           training_hooks=[logging_hook, metric_hook])


class ExportModelHook(flt.trainer_worker.ExportModelHook):
    def after_save(self, sess, model, export_dir, inputs, outputs):
        print("**************export model hook**************")
        print("sess :", sess)
        print("model: ", model)
        print("export_dir: ", export_dir)
        print("inputs: ", inputs)
        print("outpus: ", outputs)
        print("*********************************************")


if __name__ == '__main__':
    flt.trainer_worker.train(
        ROLE, args, input_fn,
        model_fn, serving_input_receiver_fn,
        export_model_hook=ExportModelHook())