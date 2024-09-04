import tenseal as ts
import numpy as np
import time
import tensorflow.compat.v1 as tf
import base64
from fedlearner.common import fl_logging


def serialize(data):
    data = convert_to_numpy(data)
    serialize_data = np.array([base64.b64encode(d.serialize()).decode('utf-8') for d in data])
    return serialize_data


def convert_to_numpy(tensor):
    if tf.is_tensor(tensor):
        tensor = tensor.numpy()
    return tensor


class TensealHEOperator:
    def __init__(self, model, tensor_type, poly_modulus_degree=4096, global_scale=2 ** 20, coeff_mod_bit_sizes=None):
        self._model = model
        # Setup TenSEAL context
        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [40, 20, 40]
        self._self_ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes
        )
        self._self_ctx.generate_galois_keys()
        self._self_ctx.global_scale = global_scale
        self._tensor_type = tensor_type

        self._other_ctx = self._self_ctx


    def encrypt(self, tensor_data, output_dim=None):
        def each_encrypt(numpy_array):
            numpy_array = convert_to_numpy(numpy_array)
            fl_logging.info(f"[Encrypt] tensor shape : {numpy_array.shape}")
            start_time = time.time()
            enc_tensor = [ts.ckks_vector(self._self_ctx, row) for row in numpy_array]
            fl_logging.info(f"[Encrypt] time: {time.time() - start_time} seconds")
            return serialize(enc_tensor)

        return tf.py_function(each_encrypt, [tensor_data], Tout=tf.string)

    def decrypt(self, tensor_data, input_dim=None, mid_dim=None, output_dim=None):
        def each_decrypt(numpy_array):
            numpy_array = self._deserialize_self(numpy_array)
            fl_logging.info(f"[Decrypt] tensor shape : ({len(numpy_array)}, {numpy_array[0].size()})")
            start_time = time.time()
            decrypted_tensor = [row.decrypt() for row in numpy_array]
            fl_logging.info(f"[Decrypt] time: {time.time() - start_time} seconds")
            # fl_logging.info(f'decrypt data : {decrypted_tensor}')
            return np.array(decrypted_tensor)

        return tf.py_function(each_decrypt, [tensor_data], Tout=self._tensor_type)

    def multiply_scalar(self, enc_x, y, input_dim=None):
        def he_multiply_scalar(enc_x, y):
            enc_x_array = self._deserialize_other(enc_x)
            fl_logging.info(
                f"[Matmul] enc_x shape : ({len(enc_x_array)}, {enc_x_array[0].size()}), y shape : {y.shape}")
            start_time = time.time()
            enc_tensor = [enc_row.matmul(y) for enc_row in enc_x_array]
            fl_logging.info(f"[Matmul] time: {time.time() - start_time} seconds")
            return serialize(enc_tensor)

        return tf.py_function(he_multiply_scalar, [enc_x, y], Tout=tf.string)

    def fhe_smm(self, enc_x, y, s, input_dim=None):
        def fhe_smm(enc_x, y, s):
            enc_x_array = self._deserialize_other(enc_x)
            fl_logging.info(
                f"[Matmul] enc_x shape : ({len(enc_x_array)}, {enc_x_array[0].size()}), y shape : {y.shape}, s shape : {s.shape}")
            start_time = time.time()
            enc_tensor = [enc_row.matmul(y) for enc_row in enc_x_array]
            fl_logging.info(f"[Matmul] time: {time.time() - start_time} seconds")
            start_time = time.time()
            result_array = []
            for i in range(0, len(enc_tensor)):
                result_array.append(enc_tensor[i] - s[i])
            fl_logging.info(f"[Subtract] time: {time.time() - start_time} seconds")
            return serialize(enc_tensor)

        return tf.py_function(fhe_smm, [enc_x, y, s], Tout=tf.string)

    def _deserialize_self(self, data):
        data = convert_to_numpy(data)
        deserialize_data = np.array([ts.ckks_vector_from(self._self_ctx, base64.b64decode(d)) for d in data])
        return deserialize_data

    def _deserialize_other(self, data):
        data = convert_to_numpy(data)
        deserialize_data = [ts.ckks_vector_from(self._other_ctx, base64.b64decode(d)) for d in data]
        return deserialize_data
