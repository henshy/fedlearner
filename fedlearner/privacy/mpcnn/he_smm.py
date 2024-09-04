import tensorflow.compat.v1 as tf

from fedlearner.privacy.mpcnn.paillier_he_operator import multiply_scalar, subtract_scalar, PaillierHEOperator, add_scalar


class PaillierSMM:

    def __init__(self, przs, model, tensor_type):
        """
        功能：基于 SMM 的 MPC NN 协同训练核心实现类
        """
        self._przs = przs
        self._model = model
        self._tensor_type = tensor_type
        self._he_operator = PaillierHEOperator(tensor_type)
        self._enc_share_w = None
        self._z2_1 = None
        self._x = None
        self._w_other_share = None

    def set_x(self, x):
        self._x = x

    def set_w_other_share(self, w_other_share):
        self._w_other_share = w_other_share

    def f_cal_dh_f(self, w1_self_share_1):
        """
        功能：非标签方计算非标签方的嵌入层梯度
        步骤：
            1. 非标签方加密己方权重秘密份额 w1_f_1，并发送给标签方
            2. 标签方收到加密的 w1_f_1 加上 非标签方分享的权重秘密份额 w1_f_2，然后和标签方输出层梯度进行乘法，得到加密的 非标签方嵌入层梯度 dhf
            3. 标签方发动加密的嵌入层梯度给非标签方进行解密，得到明文 dhf
        """
        enc_self_share_w_t_op = self._he_operator.encrypt(tf.transpose(w1_self_share_1))
        enc_dhf_op = self._model.send_and_recv('enc_f_w_t', enc_self_share_w_t_op, 'enc_dh_f')
        with tf.control_dependencies([enc_dhf_op]):
            dhf_op = self._he_operator.decrypt(enc_dhf_op)
        self._model.train_ops.append(dhf_op)
        return dhf_op

    def l_cal_dh_f(self, dy, w1_f_share_2):
        """
        功能：标签方计算非标签方的嵌入层梯度
        步骤：参考 function f_cal_dh_f
        """
        enc_f_w_t_op = self._model.recv('enc_f_w_t', tf.string, train_ops=False)
        with tf.control_dependencies([enc_f_w_t_op]):
            enc_w_t_op = add_scalar(enc_f_w_t_op, tf.transpose(w1_f_share_2))
        with tf.control_dependencies([enc_w_t_op]):
            dh_f_op = multiply_scalar(dy, enc_w_t_op)
            self._model.send('enc_dh_f', dh_f_op, depend_tensor=True)

    def l_cal_dh_l(self, dy, w1_self_share_1):
        """
        功能：标签方计算标签方的嵌入层梯度
        步骤：
            1. 非标签方加密标签方分享的权重秘密份额 w1_l_2，并发送给标签方
            2. 标签方收到加密的 w1_l_2 ，然后和标签方输出层梯度进行 SMM 乘法，得到一部分加密的标签方嵌入层梯度 enc_half_dh_l_1，并发给非标签方解密回传
            3. 标签方计算自身权重秘密份额 w1_l_1 和输出层梯度的乘积，得到另一部分嵌入层梯度
            4. 标签方对算出来的梯度进行求和，得到完整的嵌入层梯度
        """
        enc_l_share_w_op = self._model.recv('enc_l_share_w', tf.string, train_ops=False)
        with tf.control_dependencies([enc_l_share_w_op]):
            enc_half_dh_l_1_op, half_dh_l_2 = self._enc_matmul_share(dy, enc_l_share_w_op)

        half_dh_l_op = self._model.send_and_recv('enc_half_dh_l', enc_half_dh_l_1_op, 'half_dh_l', self._tensor_type)

        with tf.control_dependencies([half_dh_l_op]):
            dh_l = tf.add_n([tf.matmul(dy, w1_self_share_1, transpose_b=True), half_dh_l_op, half_dh_l_2])
        return dh_l

    def f_cal_half_dh_l(self, w1_l_share_2):
        """
        功能：非标签方计算标签方的一部分梯度
        步骤：参考 function l_cal_dh_l
        """
        enc_l_share_w_op = self._he_operator.encrypt(tf.transpose(w1_l_share_2))
        enc_half_dh_l = self._model.send_and_recv('enc_l_share_w', enc_l_share_w_op, 'enc_half_dh_l')
        with tf.control_dependencies([enc_half_dh_l]):
            half_dh_l = self._he_operator.decrypt(enc_half_dh_l)
            self._model.send('half_dh_l', half_dh_l, depend_tensor=True)

    def l_cal_gb_2(self, dy):
        """
        功能：计算非标签方权重梯度，标签方和非标签方分别得到梯度的一部分
        步骤：
            1. 标签方加密输出层梯度给非标签方
            2. 非标签方把密文梯度和自身输入特征做 SMM 矩阵乘法，得到一部分明文梯度 gb_1，并把密文部分发给标签方
            3. 标签方获取 SMM 计算得到的一部分密文梯度，并解密，得到 gb_2
        """
        with tf.control_dependencies([self._model.example_ids]):
            enc_dy_op = self._he_operator.encrypt(dy)
        with tf.control_dependencies([enc_dy_op]):
            enc_gb_2 = self._model.send_and_recv('enc_dy', enc_dy_op, 'enc_gb_2')
        with tf.control_dependencies([enc_gb_2]):
            gb_2_op = self._he_operator.decrypt(enc_gb_2)
        self._model.train_ops.append(gb_2_op)
        return gb_2_op

    def f_cal_gb_1(self, x, output_dim=None):
        """
        功能：计算非标签方权重梯度，标签方和非标签方分别得到梯度的一部分
        步骤：参考 function l_cal_gb_2
        """
        enc_dy_op = self._model.recv('enc_dy', tf.string, train_ops=False)
        enc_gb_2, gb_1 = self._enc_matmul_share(tf.transpose(x), enc_dy_op)
        self._model.send('enc_gb_2', enc_gb_2, depend_tensor=True)
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

    def exchange_enc_data(self):
        enc_other_share_w_op = self._he_operator.encrypt(self._w_other_share)
        self._enc_share_w = self._model.send_and_recv('enc_other_share_w', enc_other_share_w_op)

    def cal_enc_z2(self):
        enc_z2_2, self._z2_1 = self._enc_matmul_share(self._x, self._enc_share_w)
        return enc_z2_2

    def get_share_z2_2(self, enc_z2_2):
        enc_share_z2_2 = self._model.send_and_recv('enc_z2_2', enc_z2_2)
        with tf.control_dependencies([enc_share_z2_2]):
            share_z2_2_op = self._he_operator.decrypt(enc_share_z2_2)
        return share_z2_2_op

    def cal_self_result(self, share_z2_2, w1_self_share_1):
        z1 = tf.matmul(self._x, w1_self_share_1)
        with tf.control_dependencies([share_z2_2]):
            self_result = tf.add_n([z1, self._z2_1, share_z2_2])
        return self_result

    def _enc_matmul_share(self, mat_a, enc_mat_b):
        """
        功能：完成密文矩阵和明文矩阵乘法的秘密分享，mat_a * enc_mat_b = mat_c + enc_mat_d
        输入：一个密文矩阵和一个明文矩阵完成矩阵乘法
        输出：一个明文的随机矩阵和一个密文矩阵
        """
        with tf.control_dependencies([mat_a, enc_mat_b]):
            enc_mat_op = multiply_scalar(mat_a, enc_mat_b)
        with tf.control_dependencies([enc_mat_op]):
            self_share_1 = self._przs.generate_random_tensor(tf.shape(enc_mat_op))
        with tf.control_dependencies([self_share_1]):
            enc_share_2_op = subtract_scalar(enc_mat_op, self_share_1)
        return enc_share_2_op, self_share_1
