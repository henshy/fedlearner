import numpy as np
import tensorflow.compat.v1 as tf

from fedlearner.common import fl_logging
from fedlearner.privacy.mpcnn.he_smm import PaillierSMM
from fedlearner.privacy.mpcnn.przs import PRZS
from fedlearner.privacy.mpcnn.fhe_smm import SealSMM


def py_func_grad(func, inp, Tout, stateful=True, name=None, grad=None):
    """
    功能：实现 MPC NN 层的自义定梯度计算
    说明：使用 py_func 封装正向传播逻辑，使用 RegisterGradient 注册自定义梯度，再使用 gradient_override_map 覆盖正向传播的默认梯度逻辑
    """
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name, "PyFuncStateless": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)


class NNMPCLayer(tf.layers.Layer):

    def __init__(self, batch_size, output_dim, other_party_input_dim, model, mode, tensor_type=tf.float32, glorot_initializer=True, label_role='leader', he_method='CKKS', use_gpu=False):
        """
        功能：两方纵向 NN 的浅层 MPC 协同训练，嵌入层本地训练，嵌入的下一层通过 MPC 协同训练，之后的模型在标签方进行训练
        参数说明：
            output_dim：MPC 交互层输出的权重维度（非 shape）
            other_party_input_dim：对方 MPC 层输入层的维度（非 shape）
            model：封装 FL 的联邦 model
            mode：train、eval、predict
            tensor_type：MPC NN 层的 tensor 精度，默认为 float32
            glorot_initializer：为防止梯度消失或梯度爆炸，使用 glorot 初始化 权重 和 秘密分享 tensor 矩阵，默认为 True
            label_role：标签方在哪个角色运行，默认是 leader
        """
        super(NNMPCLayer, self).__init__()
        self._batch_size = batch_size
        self._output_dim = output_dim
        self._glorot_initializer = glorot_initializer
        self._model = model
        self._mode = mode
        self._label_role = label_role
        self._other_input_dim = other_party_input_dim
        self._tensor_type = tensor_type
        self._he_method = he_method
        self._other_w_shape = None
        self._self_w_shape = None
        self._w1_other_share = None
        self._w1_self_share = None
        self._w1_mpc = None
        self._batchs = None
        self._steps = None
        self._przs = None
        self._smm = None
        self._use_gpu = use_gpu

    def build(self, input_shape):
        """ 功能：通过特征shape 和 输入维度，初始化秘密分享的权重矩阵，为实际训练做准备 """
        self._self_w_shape = [int(input_shape[-1]), self._output_dim]
        self._other_w_shape = [self._other_input_dim, self._output_dim]
        fl_logging.info(
            f'Init NNMPCLayer Weight, input shape: {input_shape}, '
            f'self weight shape: {self._self_w_shape}, '
            f'other party weight shape: {self._other_w_shape}')
        if self._mode == tf.estimator.ModeKeys.TRAIN or self._mode == tf.estimator.ModeKeys.EVAL:
            self._przs = PRZS(self._self_w_shape, self._other_w_shape, self._model, self._glorot_initializer)
            if self._he_method == 'CKKS':
                self._smm = SealSMM(self._przs, self._model, self._tensor_type, self._use_gpu)
            elif self._he_method == 'Paillier':
                self._smm = PaillierSMM(self._przs, self._model, self._tensor_type)
            self._przs.run_przs()
            self._generate_weight()
        else:
            self._w1_mpc = tf.get_variable(name='w_mpc', shape=self._self_w_shape, initializer=tf.zeros_initializer(),
                                           trainable=True)
        super(NNMPCLayer, self).build(input_shape)

    def _generate_weight(self):
        """ 功能：初始化秘密分享权重矩阵，第三个权重做为最终秘密分享还原的权重 """
        self._w1_self_share = tf.get_variable(name='w_self_share_1', initializer=self._przs.self_share_1, trainable=True)
        self._w1_other_share = tf.get_variable(name='w_other_share_2', initializer=self._przs.other_share_2, trainable=True)
        self._w1_mpc = tf.get_variable(name='w_mpc', shape=self._self_w_shape, initializer=tf.zeros_initializer(), trainable=True)

        # 初始化 _batches 为 0
        self._batches = tf.get_variable(name='batches', initializer=tf.zeros([1, 1], dtype=tf.int32), trainable=True)

    def call(self, inputs):
        """
        功能：进行 MPC NN 层交互计算
        说明：
            1. 如果是训练，正向交互计算得到交互层结果，反向需要通过自定义的梯度方法计算得到权重梯度和嵌入层梯度
            2. 如果是 eval，走 MPC 的正向传播流程即可
            3. 如果是 predict，计算单边结果，激活第三权重，导出模型，为在线推理服务
        """
        if self._mode == tf.estimator.ModeKeys.TRAIN:
            z = py_func_grad(self._forward, inp=[inputs, self._w1_self_share, self._w1_other_share], Tout=self._tensor_type, grad=self.custom_nn_mpc_layer_grad)
            with tf.control_dependencies([z]):
                self._w_restore()
        elif self._mode == tf.estimator.ModeKeys.EVAL:
            z = self._forward(inputs, self._w1_self_share, self._w1_other_share)
        else:
            z = tf.matmul(inputs, self._w1_mpc)
        return z

    def _forward(self, x, w1_self_share, w1_other_share):
        """
        功能：进行 MPC NN 正向传播
        步骤：
            1. 由于采用 py_func 封装，输入先转化为 tensor
            2. 加密属于对方的秘密份额权重给对方，在密文下完成 SMM 计算，并把 SMM 计算得到的密文结果回传所属方解密，各方得到一部分明文结果
            3. 非标签方发送计算出来的明文结果给标签方，标签方汇总得到最终结果
        """
        fl_logging.info(f"[Forward] init")
        x = tf.convert_to_tensor(x)
        w1_self_share = tf.convert_to_tensor(w1_self_share)
        w1_other_share = tf.convert_to_tensor(w1_other_share)
        self._smm.set_x(x)
        self._smm.set_w_other_share(w1_other_share)
        fl_logging.info(f"[Forward] exchange_enc_data")
        self._smm.exchange_enc_data(self._output_dim)

        fl_logging.info(f"[Forward] cal_enc_z2")
        enc_z2_2 = self._smm.cal_enc_z2(self._batch_size)
        share_z2_2 = self._smm.get_share_z2_2(enc_z2_2)
        self_result = self._smm.cal_self_result(share_z2_2, w1_self_share)
        if self._model.role == self._label_role:
            with tf.control_dependencies([self_result]):
                other_result = self._model.recv('self_result', self_result.dtype, depend_tensor=True)
            z = tf.add(other_result, self_result)
        else:
            self._model.send('self_result', self_result, depend_tensor=True)
            z = self_result
        fl_logging.info(f"[Forward] Done")
        return z

    def custom_nn_mpc_layer_grad(self, op, grad):
        """
        功能：进行 MPC NN 层的自定义反向传播，计算得到嵌入层梯度和秘密份额权重的梯度
        步骤：
            1. 取出秘密份额的权重，在 SMM 下计算得到嵌入层梯度
            2. 取出输入层特征，在 SMM 下计算得到权重梯度
        """
        w1_self_share = op.inputs[1]
        w1_other_share = op.inputs[2]
        x = op.inputs[0]
        if self._model.role == self._label_role:
            self._smm.l_cal_dh_f(grad, w1_other_share)
            dh_l = self._smm.l_cal_dh_l(grad, w1_self_share)

            gb_2 = self._smm.l_cal_gb_2(grad, self._other_input_dim)
            ga_1 = self._smm.cal_ga_1(x, grad)
            return dh_l, ga_1, gb_2
        else:
            dh_f = self._smm.f_cal_dh_f(w1_self_share, self._batch_size)
            self._smm.f_cal_half_dh_l(w1_other_share, self._batch_size)

            gb_1 = self._smm.f_cal_gb_1(x, self._output_dim)
            ga_2 = self._smm.get_ga_2()
            return dh_f, gb_1, ga_2

    def _w_restore(self):
        """
        功能：为了减少在线 serving 的延迟，计算完后，还原参与方的 MPC 层的权重
        步骤：
            1. 获取总的 batch 数以及当前训练到的 batch，当训练到最后一个 batch 时，还原参与方的 MPC 层权重
            2. 采用第三个权重做为最终权重的方式，其他两个权重训练完不变，可以支持增量更新
        """
        restore_w1_mpc = tf.py_func(self._pyfunc_w_restore,
                                        [self._w1_mpc, self._w1_other_share, self._w1_self_share, self._batches],
                                        Tout=self._tensor_type)
        with tf.control_dependencies([restore_w1_mpc]):
            update_op = tf.assign(self._w1_mpc, restore_w1_mpc)
            self._model.train_ops.append(update_op)
            updated_batches = tf.py_func(self._pyfunc_update_batches, [self._batches], Tout=tf.int32)
        with tf.control_dependencies([updated_batches]):
            update_batches_op = tf.assign(self._batches, updated_batches)
            self._model.train_ops.append(update_batches_op)

    def _pyfunc_update_batches(self, batches):
        batch_num = self._model.batch_num
        # 更新 batches
        batches[0] = batches[0] + 1
        fl_logging.info(f'now batches are : {batches[0]}, batch_num : {batch_num}')
        # 条件判断并更新 batches
        if batches[0] != 0 and batch_num != 0 and batches[0] == batch_num:
            batches[0] = 0
        return batches

    def _pyfunc_w_restore(self, w1_mpc, w1_other_share, w1_self_share, batches):
        batch_num = self._model.batch_num
        # 更新 batches
        batches[0] = batches[0] + 1
        fl_logging.info(f'now batches are : {batches[0]}, batch_num : {batch_num}')
        # 条件判断并更新 w1_mpc
        if batches[0] != 0 and batch_num != 0 and batches[0] == batch_num:
            w1_self_share = tf.convert_to_tensor(w1_self_share)
            w1_mpc = tf.add(self._model.send_and_recv('w_other_share_2', w1_other_share), w1_self_share)
        return w1_mpc
