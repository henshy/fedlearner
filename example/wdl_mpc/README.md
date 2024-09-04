# 单机执行测试
### 1. 先打包成 docker 镜像
git clone https://github.com/henshy/fedlearner.git
cd fedlearner
git checkout gix_dev_sgx
cd sgx
bash -x build_dev_docker_image.sh
bash -x build_release_docker_image.sh

### 2. 启动容器镜像（如果要用 GPU，请使用 nvidia docker，并挂载 GPU启动）
docker run -it fedlearner-sgx-dev:latest bash

### 3. 在另一个窗口进入刚刚启动的容器（docker ps 找到对应的 ps id）
docker exec -it 容器ID bash

### 4. 进入 example 路径，做准备工作
cd /usr/local/lib/python3.6/dist-packages/example/wdl_mpc/
### 4.1 创建测试数据
python make_data_mpc.py
### 4.2 把全同态和tensorflow io依赖包放入当前路径
cp /troy-nova/pybind/pytroy_raw.cpython-36m-x86_64-linux-gnu.so ./
cp /gramine/CI-Examples/tensorflow_io.py ./

### 5. 进行代码测试
#### 注意：如需要使用 GPU，需要修改算法代码，传入 use_gpu=True 参数，并在对应窗口分别指定任务使用的 GPU 卡，可通过 export CUDA_VISIBLE_DEVICES=0 设置（代表使用第一个 GPU），follower 可设置 export CUDA_VISIBLE_DEVICES=1
### 5.1 leader 执行如下命令
python leader_mpc.py --local-addr=localhost:50051 --peer-addr=localhost:50052 --data-path=data/leader --checkpoint-path=log/leader/checkpoint --save-checkpoint-steps=10 --summary-path=log/leader/summary --summary-save-steps=10 --batch-size=128 '--data-path-wildcard=**/*.tfrecords' --self-seed=-7480604591810883409 --other-seed=-3484824100278629883 --loglevel=debug --num-example=2000
### 5.2 follower 在另一个窗口执行如下命令
python follower_mpc.py --local-addr=localhost:50052 --peer-addr=localhost:50051 --data-path=data/leader --checkpoint-path=log/follower/checkpoint --save-checkpoint-steps=10 --summary-path=log/follower/summary --summary-save-steps=10 --batch-size=128 '--data-path-wildcard=**/*.tfrecords' --loglevel=debug --self-seed=-3484824100278629883 --other-seed=-7480604591810883409 --num-example=2000

### 注意：每次执行完需要删除当前路径下的 log 路径，不然会加载之前的模型进行训练
