import numpy as np
import tensorflow.compat.v1 as tf

from fedlearner.common import fl_logging
from fedlearner.model.crypto import fate_paillier as paillier
import pickle


def serialize(data):
    data = convert_to_numpy(data)
    serialize_data = np.array([pickle.dumps(d) for d in data.flatten()])
    return serialize_data.reshape(data.shape)


def deserialize(data):
    data = convert_to_numpy(data)
    deserialize_data = np.array([pickle.loads(d) for d in data.flatten()])
    return deserialize_data.reshape(data.shape)


def convert_to_numpy(tensor):
    if tf.is_tensor(tensor):
        tensor = tensor.numpy()
    return tensor


def multiply_scalar(x, enc_w):
    def he_multiply_scalar(x, enc_w):
        enc_w_array = deserialize(enc_w)
        result_array = np.dot(x, enc_w_array)
        return serialize(result_array)

    return tf.py_function(he_multiply_scalar, [x, enc_w], Tout=tf.string)


def subtract_scalar(enc_a, b):
    def he_subtract_scalar(enc_x, y):
        enc_x_array = deserialize(enc_x)
        result_array = np.subtract(enc_x_array, y)
        return serialize(result_array)

    return tf.py_function(he_subtract_scalar, [enc_a, b], Tout=tf.string)


def add_scalar(enc_a, b):
    def he_add_scalar(enc_x, y):
        enc_x_array = deserialize(enc_x)
        result_array = np.add(enc_x_array, y)
        return serialize(result_array)

    return tf.py_function(he_add_scalar, [enc_a, b], Tout=tf.string)


def print_tensor(x, name):
    def print_func(x):
        fl_logging.info(f"{name} : {x}")
        return x

    return tf.py_function(print_func, [x], Tout=x.dtype)


class PaillierHEOperator:
    def __init__(self, tensor_type):
        self._public_key, self._private_key = paillier.PaillierKeypair().generate_keypair()
        self._tensor_type = tensor_type

    @property
    def tensor_type(self):
        return self._tensor_type

    def encrypt(self, tensor_data):
        def each_encrypt(numpy_array):
            numpy_array = convert_to_numpy(numpy_array)
            fl_logging.info(f'encrypt shape : {numpy_array.shape}')
            fl_logging.info(f'encrypt : {numpy_array[0]}')
            encrypted_array = np.array(
                [self._public_key.encrypt(val) for val in numpy_array.flatten()])
            encrypted_array = encrypted_array.reshape(numpy_array.shape)
            return serialize(encrypted_array)

        return tf.py_function(each_encrypt, [tensor_data], Tout=tf.string)

    def decrypt(self, tensor_data):
        def each_decrypt(numpy_array):
            numpy_array = deserialize(numpy_array)
            np.set_printoptions(precision=None, suppress=True)
            decrypted_array = np.array(
                [self._private_key.decrypt(val) for val in numpy_array.flatten()])
            decrypted_array = decrypted_array.reshape(numpy_array.shape)
            fl_logging.info(f'decrypt data : {decrypted_array}')
            return decrypted_array

        return tf.py_function(each_decrypt, [tensor_data], Tout=self._tensor_type)
