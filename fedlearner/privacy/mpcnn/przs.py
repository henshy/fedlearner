import tensorflow.compat.v1 as tf
import os


class PRZS:
    def __init__(self, w_shape, w_other_shape, model, glorot_initializer, tensor_type=tf.float32, minval=-0.1, maxval=0.1):
        """
        功能：
            1. 通过分享随机种子，实现权重本地秘密分享，无需交换秘密份额权重
            2. 实现 glorot 和指定范围的 tensor 初始化
        """
        self._minval = minval
        self._maxval = maxval
        self._tensor_type = tensor_type
        self._glorot_initializer = glorot_initializer
        self._w = self.generate_random_tensor(w_shape)
        self._model = model
        self._self_w_shape = w_shape
        self._self_w_seed = self._model.self_seed
        self._other_w_shape = w_other_shape
        self._other_w_seed = self._model.other_seed
        self._self_w_share_1 = None
        self._other_w_share_2 = None

    @property
    def other_share_2(self):
        return self._other_w_share_2

    @property
    def self_share_1(self):
        return self._self_w_share_1

    def generate_random_tensor(self, shape):
        return self._generate_random_ring_element(shape, int.from_bytes(os.urandom(8), "big") - 2 ** 63)

    def _generate_random_ring_element(self, shape, seed):
        if seed:
            if self._glorot_initializer:
                n = tf.cast(shape[0] + shape[1], self._tensor_type)
                stddev = tf.sqrt(2. / n)
                return tf.random.normal(shape, mean=0., stddev=stddev, seed=int(seed), dtype=self._tensor_type)
            return tf.random.uniform(
                shape=shape, minval=self._minval, maxval=self._maxval, seed=int(seed), dtype=self._tensor_type)
        else:
            return self.generate_random_tensor(shape)

    def _przs_other(self):
        self._other_w_share_2 = self._przs_share(self._other_w_shape, self._self_w_seed, self._other_w_seed)

    def _przs_self(self):
        share = self._przs_share(self._self_w_shape, self._other_w_seed, self._self_w_seed)
        self._self_w_share_1 = tf.subtract(self._w, share)
        self._self_w_share_2 = share

    def _przs_share(self, shape, seed1, seed2, device=None):
        device = '/cpu:0' if device is None else device
        with tf.device(device):
            current_share = self._generate_random_ring_element(shape, seed1)
            next_share = self._generate_random_ring_element(shape, seed2)
            share = current_share - next_share
            return share

    def run_przs(self):
        self._przs_other()
        self._przs_self()
