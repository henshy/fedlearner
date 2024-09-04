import numpy as np
import tensorflow.compat.v1 as tf

from fedlearner.privacy.mpcnn.seal_he_operator import SealHEOperator
from fedlearner.privacy.mpcnn.tenseal_he_operator import TensealHEOperator


class SealSMM:

    def __init__(self, przs, model, tensor_type, use_gpu):
        """
        功能：基于 SMM 的 MPC NN 协同训练核心实现类
        """
        self._przs = przs
        self._model = model
        self._tensor_type = tensor_type
        # self._he_operator = TensealHEOperator(model, tensor_type)
        self._use_gpu = use_gpu
        self._he_operator = SealHEOperator(model, tensor_type, self._use_gpu)
        self._z2_1 = None
        self._x = None
        self._w = None
        self._enc_other_x = None
        self._w_other_share = None
        self._enc_dy = None

    def set_x(self, x):
        self._x = x

    def set_w_other_share(self, w_other_share):
        self._w_other_share = w_other_share

    def f_cal_dh_f(self, w1_self_share_1, input_dim=None):
        """
        功能：非标签方计算非标签方的嵌入层梯度
        步骤：
            1. 标签方加密己方梯度秘密 dy，并发送给非标签方
            2. 非标签方收到加密的 dy 和己方权重秘密份额 w1_f_1 进行 SMM 乘法，得到一部分加密的嵌入层梯度 enc_half_dh_f_1 并发给标签方进行解密
            3. 标签方收到解密 enc_half_dh_f_1，并计算明文梯度和非标签方分享的权重秘密份额乘积 half_df_f_l，再把 enc_half_dh_f_1 与 half_df_f_l 求和后传给非标签方
            3. 非标签方收到 half_dh_f_2_l，并和己方之前进行 SMM 计算得到的明文 half_dh_f_2 求和，得到嵌入层梯度 dhf
        """
        self._enc_dy = self._model.recv('enc_dy', tf.string, train_ops=False)
        enc_half_dh_f_1_op, half_dh_f_2 = self._enc_matmul_share(self._enc_dy, tf.transpose(w1_self_share_1), input_dim)

        half_dh_f_2_l = self._model.send_and_recv('enc_half_dh_f_1', enc_half_dh_f_1_op, 'half_dh_f_2_l',
                                                  self._tensor_type)
        with tf.control_dependencies([half_dh_f_2_l]):
            return tf.add(half_dh_f_2, half_dh_f_2_l)

    def l_cal_dh_f(self, dy, w1_f_share_2):
        """
        功能：标签方计算非标签方的嵌入层梯度
        步骤：参考 function f_cal_dh_f
        """
        half_df_f_l = tf.matmul(dy, w1_f_share_2, transpose_b=True)
        dh_f_shape = tf.shape(half_df_f_l)
        enc_dy = self._he_operator.encrypt(dy, dh_f_shape[1])
        enc_half_dh_f_1 = self._model.send_and_recv('enc_dy', enc_dy, 'enc_half_dh_f_1')

        with tf.control_dependencies([enc_half_dh_f_1]):
            half_dh_f_1_op = self._he_operator.decrypt(enc_half_dh_f_1, dh_f_shape[0], tf.shape(dy)[1], dh_f_shape[1])

        with tf.control_dependencies([half_dh_f_1_op]):
            half_dh_f_2_l = tf.add(half_dh_f_1_op, half_df_f_l)
            self._model.send('half_dh_f_2_l', half_dh_f_2_l, depend_tensor=True)

    def l_cal_dh_l(self, dy, w1_self_share_1):
        """
        功能：标签方计算标签方的嵌入层梯度
        步骤：
            1. 标签方加密己方梯度秘密 dy，并发送给非标签方
            2. 非标签方收到加密的 dy 和标签方权重秘密份额 w1_l_share_2 进行 SMM 乘法，得到一部分加密的嵌入层梯度 enc_dh_l_f_2，以及明文嵌入层梯度 half_dh_l，共同发给标签方进行计算
            3. 标签方收到解密 enc_dh_l_f_2，并计算明文梯度和己方权重秘密份额 w1_self_share_1 乘积得到 dh_l_l
            3. 标签方最后加和 dh_l_f_2、dh_l_l、half_dh_l，得到嵌入层梯度 dhl
        """
        w1_self_share_1_t = tf.transpose(w1_self_share_1)
        dh_l_l = tf.matmul(dy, w1_self_share_1_t)
        dh_l_shape = tf.shape(dh_l_l)

        enc_dy = self._he_operator.encrypt(dy, dh_l_shape[1])
        enc_dh_l_f_2 = self._model.send_and_recv('enc_dy_l', enc_dy, 'enc_dh_l_f_2')
        with tf.control_dependencies([enc_dh_l_f_2]):
            dh_l_f_2 = self._he_operator.decrypt(enc_dh_l_f_2, dh_l_shape[0], tf.shape(w1_self_share_1_t)[0],
                                                 dh_l_shape[1])

        with tf.control_dependencies([dh_l_f_2]):
            dh_l = tf.add(dh_l_l, dh_l_f_2)
        return dh_l

    def f_cal_half_dh_l(self, w1_l_share_2, input_dim=None):
        """
        功能：非标签方计算标签方的一部分梯度
        步骤：参考 function l_cal_dh_l
        """
        enc_dy = self._model.recv('enc_dy_l', tf.string, train_ops=False)
        with tf.control_dependencies([enc_dy]):
            enc_dh_l_f_2 = self._he_operator.multiply_scalar(enc_dy, tf.transpose(w1_l_share_2), input_dim)
            self._model.send('enc_dh_l_f_2', enc_dh_l_f_2, depend_tensor=True)

    def l_cal_gb_2(self, dy, input_dim=None):
        """
        功能：计算非标签方权重梯度，标签方和非标签方分别得到梯度的一部分
        步骤：
            1. 非标签方加密输出层 EMB 给非标签方
            2. 标签方把密文 EMB 和梯度做 SMM 矩阵乘法，得到一部分明文梯度 gb_1，并把密文部分发给非标签方
            3. 非标签方获取 SMM 计算得到的一部分密文梯度，并解密，得到 gb_2
        """
        enc_x = self._model.recv('enc_x', tf.string, train_ops=False)
        enc_gb_1, gb_1 = self._enc_matmul_share(enc_x, dy, input_dim)
        self._model.send('enc_gb_1', enc_gb_1, depend_tensor=True)
        return gb_1

    def f_cal_gb_1(self, x, output_dim=None):
        """
        功能：计算非标签方权重梯度，标签方和非标签方分别得到梯度的一部分
        步骤：参考 function l_cal_gb_2
        """
        x_trans = tf.transpose(x)
        x_trans_shape = tf.shape(x_trans)
        enc_x = self._he_operator.encrypt(x_trans, output_dim)
        enc_gb_1 = self._model.send_and_recv('enc_x', enc_x, 'enc_gb_1')
        with tf.control_dependencies([enc_gb_1]):
            gb_1 = self._he_operator.decrypt(enc_gb_1, x_trans_shape[0], x_trans_shape[1], output_dim)
        self._model.train_ops.append(gb_1)
        return gb_1

    def cal_ga_1(self, x, dy):
        """
        功能：计算标签方权重梯度，标签方和非标签方分别得到梯度的一部分
        步骤：标签方计算明文的特征和输出梯度的矩阵乘法，得到完整梯度，再通过秘密分享给到非标签方，各方拥有一部分明文梯度
        """
        ga = tf.matmul(x, dy, transpose_a=True)
        ga_2 = self._przs.generate_random_tensor(tf.shape(ga))
        self._model.send('ga_2', ga_2)
        ga_1 = tf.subtract(ga, ga_2)
        return ga_1

    def get_ga_2(self):
        ga_2 = self._model.recv('ga_2')
        return ga_2

    def exchange_enc_data(self, output_dim=None):
        enc_self_x_op = self._he_operator.encrypt(self._x, output_dim)
        self._enc_other_x = self._model.send_and_recv('enc_self_x', enc_self_x_op)

    def cal_enc_z2(self, input_dim=None):
        enc_z2_2, self._z2_1 = self._enc_matmul_share(self._enc_other_x, self._w_other_share, input_dim)
        return enc_z2_2

    def get_share_z2_2(self, enc_z2_2):
        enc_share_z2_2 = self._model.send_and_recv('enc_z2_2', enc_z2_2)
        enc_share_z2_shape = tf.shape(self._z2_1)
        with tf.control_dependencies([enc_share_z2_2]):
            share_z2_2_op = self._he_operator.decrypt(enc_share_z2_2, enc_share_z2_shape[0],
                                                      tf.shape(self._w_other_share)[0], enc_share_z2_shape[1])
        return share_z2_2_op

    def cal_self_result(self, share_z2_2, w1_self_share_1):
        z1 = tf.matmul(self._x, w1_self_share_1)
        with tf.control_dependencies([share_z2_2]):
            self_result = tf.add_n([z1, self._z2_1, share_z2_2])
        return self_result

    def _enc_matmul_share(self, enc_mat_a, mat_b, input_dim=None):
        """
        功能：完成密文矩阵和明文矩阵乘法的秘密分享，enc_mat_a * mat_b = mat_c + enc_mat_d
        输入：一个密文矩阵和一个明文矩阵完成矩阵乘法
        输出：一个明文的随机矩阵和一个密文矩阵
        """
        mat_b_shape = tf.shape(mat_b)
        enc_mat_shape = tf.convert_to_tensor([input_dim, mat_b_shape[1]])
        self_share_1 = self._przs.generate_random_tensor(enc_mat_shape)
        with tf.control_dependencies([enc_mat_a, mat_b]):
            enc_share_2_op = self._he_operator.fhe_smm(enc_mat_a, mat_b, self_share_1, input_dim)
        return enc_share_2_op, self_share_1
