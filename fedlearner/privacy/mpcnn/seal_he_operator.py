import numpy as np
import time
import tensorflow.compat.v1 as tf
import pytroy_raw as pytroy
from pytroy_raw import SchemeType
import numpy as np
import base64
from fedlearner.common import fl_logging
from fedlearner.privacy.mpcnn.seal_utils import GeneralHeContext


def convert_to_numpy(tensor):
    if tf.is_tensor(tensor):
        tensor = tensor.numpy()
    return tensor


class SealHEOperator:
    def __init__(self, model, tensor_type, use_gpu=False, poly_modulus_degree=4096, global_scale=20,
                 coeff_mod_bit_sizes=None, pack_lwe=False):
        self._model = model
        self._tensor_type = tensor_type
        self._pack_lwe = pack_lwe

        if coeff_mod_bit_sizes is None:
            coeff_mod_bit_sizes = [40, 20, 40]

        self._ghe = GeneralHeContext(use_gpu, SchemeType.CKKS, poly_modulus_degree, global_scale, coeff_mod_bit_sizes,
                                     True, 0x123, 5, 1 << 20, 1e-2)

        self._he = self._ghe.context
        self._encoder = self._ghe.encoder
        self._encryptor = self._ghe.encryptor
        self._decryptor = self._ghe.decryptor
        self._evaluator = self._ghe.evaluator

    def encrypt(self, tensor_data, output_dim):
        def each_encrypt(numpy_array, output_dim):
            numpy_array = convert_to_numpy(numpy_array)
            helper = pytroy.MatmulHelper(numpy_array.shape[0], numpy_array.shape[1], output_dim,
                                         self._ghe.parms.poly_modulus_degree(),
                                         pytroy.MatmulObjective.EncryptLeft, self._pack_lwe)
            data_encoded = helper.encode_inputs_doubles(self._encoder.encoder, numpy_array.flatten(), None,
                                                        self._ghe.scale)
            fl_logging.info(f"[Encrypt] tensor shape : {numpy_array.shape}")
            start_time = time.time()
            data_encrypted = data_encoded.encrypt_symmetric(self._encryptor)

            fl_logging.info(f"[Encrypt] time: {time.time() - start_time} seconds")
            return self._serialize_input(data_encrypted)

        return tf.py_function(each_encrypt, [tensor_data, output_dim], Tout=tf.string)

    def decrypt(self, tensor_data, input_dim, mid_dim, output_dim):
        def each_decrypt(numpy_array, input_dim, mid_dim, output_dim):
            fl_logging.info(f"[Decrypt] tensor shape : ({input_dim}, {output_dim})")
            numpy_array = convert_to_numpy(numpy_array)
            helper = pytroy.MatmulHelper(input_dim, mid_dim, output_dim,
                                         self._ghe.parms.poly_modulus_degree(),
                                         pytroy.MatmulObjective.EncryptLeft, self._pack_lwe)
            deserialize_data = helper.deserialize_outputs(self._evaluator, base64.b64decode(numpy_array))
            start_time = time.time()
            decrypted_data = helper.decrypt_outputs_doubles(self._encoder.encoder, self._decryptor, deserialize_data)
            fl_logging.info(f"[Decrypt] time: {time.time() - start_time} seconds")
            # fl_logging.info(f'decrypt data : {decrypted_tensor}')
            return np.reshape(decrypted_data, (input_dim, output_dim))

        return tf.py_function(each_decrypt, [tensor_data, input_dim, mid_dim, output_dim], Tout=self._tensor_type)

    def add_scalar(self, enc_a, b):
        def he_add_scalar(enc_x, y):
            enc_x_array = self._deserialize(enc_x)
            helper = pytroy.MatmulHelper(enc_x_array.rows(), 0, enc_x_array.columns(),
                                         self._ghe.parms.poly_modulus_degree(),
                                         pytroy.MatmulObjective.EncryptLeft, self._pack_lwe)
            y_encoded = helper.encode_outputs_doubles(self._encoder.encoder, y.flatten(), None,
                                                      self._ghe.scale * self._ghe.scale)
            enc_x_array.add_plain_inplace(self._evaluator, y_encoded)
            return self._serialize(helper, enc_x_array)

        return tf.py_function(he_add_scalar, [enc_a, b], Tout=tf.string)

    def fhe_smm(self, enc_x, y, s, input_dim=None):
        def fhe_smm(enc_x, y, s, input_dim):
            enc_x_array = self._deserialize(enc_x)
            y = convert_to_numpy(y)
            s = convert_to_numpy(s)
            helper = pytroy.MatmulHelper(input_dim, y.shape[0], y.shape[1],
                                         self._ghe.parms.poly_modulus_degree(),
                                         pytroy.MatmulObjective.EncryptLeft, self._pack_lwe)
            y_encoded = helper.encode_weights_doubles(self._encoder.encoder, y.flatten(), None, self._ghe.scale)
            s_encoded = helper.encode_outputs_doubles(self._encoder.encoder, s.flatten(), None,
                                                      self._ghe.scale * self._ghe.scale)
            fl_logging.info(
                f"[Matmul] enc_x shape : ({input_dim}, {y.shape[0]}), y shape : {y.shape}")
            start_time = time.time()
            enc_tensor = helper.matmul(self._evaluator, enc_x_array, y_encoded)
            fl_logging.info(f"[Matmul] time: {time.time() - start_time} seconds")
            start_time = time.time()
            enc_tensor.sub_plain_inplace(self._evaluator, s_encoded)
            fl_logging.info(f"[Subtract] time: {time.time() - start_time} seconds")
            return self._serialize(helper, enc_tensor)

        return tf.py_function(fhe_smm, [enc_x, y, s, input_dim], Tout=tf.string)

    def multiply_scalar(self, enc_x, y, input_dim=None):
        def fhe_multiply_scalar(enc_x, y, input_dim):
            enc_x_array = self._deserialize(enc_x)
            y = convert_to_numpy(y)
            helper = pytroy.MatmulHelper(input_dim, y.shape[0], y.shape[1],
                                         self._ghe.parms.poly_modulus_degree(),
                                         pytroy.MatmulObjective.EncryptLeft, self._pack_lwe)
            y_encoded = helper.encode_weights_doubles(self._encoder.encoder, y.flatten(), None, self._ghe.scale)
            fl_logging.info(
                f"[Matmul] enc_x shape : ({input_dim}, {y.shape[0]}), y shape : {y.shape}")
            start_time = time.time()
            enc_tensor = helper.matmul(self._evaluator, enc_x_array, y_encoded)
            fl_logging.info(f"[Matmul] time: {time.time() - start_time} seconds")
            return self._serialize(helper, enc_tensor)

        return tf.py_function(fhe_multiply_scalar, [enc_x, y, input_dim], Tout=tf.string)

    def _serialize(self, helper, data):
        data_serialized = helper.serialize_outputs(self._evaluator, data)
        serialize_data = np.array(base64.b64encode(data_serialized).decode('utf-8'))
        return serialize_data

    def _serialize_input(self, data):
        data_serialized = data.save(self._he)
        serialize_data = np.array(base64.b64encode(data_serialized).decode('utf-8'))
        # fl_logging.info(f"[Serialize] serialize_data: {serialize_data}")
        return serialize_data

    def _deserialize(self, data):
        data = convert_to_numpy(data)
        # fl_logging.info(f"[Deserialize] deserialize_data: {data}")
        deserialize_data = pytroy.Cipher2d.load_new(base64.b64decode(data), self._he)
        return deserialize_data
