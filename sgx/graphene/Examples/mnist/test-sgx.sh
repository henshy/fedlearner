#!/bin/bash
set -e
set -x

shopt -s expand_aliases
alias logfilter="grep -v \"FUTEX\|measured\|memory entry\|cleaning up\|async event\|shim_exit\""

export CUDA_VISIBLE_DEVICES=""

custom_env="custom_env"
function get_env() {
    graphene-sgx-get-token -sig=python.sig  | grep $1 | awk -F":" '{print $2}' | xargs
}

function make_custom_env() {
    echo "TF_OPTIONAL_TLS_ENABLE=on" > $custom_env 
    echo "MR_ENCLAVE=`get_env mr_enclave`" >> $custom_env
    echo "MR_SIGNER=`get_env mr_signer`" >> $custom_env
    echo "ISV_PROD_ID=`get_env isv_prod_id`" >> $custom_env
    echo "ISV_SVN=`get_env isv_svn`" >> $custom_env
    # make no sense right now
    echo "DEBUG=0" >> $custom_env
    echo "parallel_num_threads=2" >> $custom_env
    echo "session_parallelism=0" >> $custom_env
    echo "intra_op_parallelism=2" >> $custom_env
    echo "inter_op_parallelism=2" >> $custom_env
    echo "OMP_NUM_THREADS=2" >> $custom_env
    echo "MKL_NUM_THREADS=2" >> $custom_env
}


ROLE=$1
if [ "$ROLE" == "data" ]; then
    make_custom_env
    rm -rf data
    python make_data.py
fi

if [ "$ROLE" == "leader" ]; then
    rm -rf model/leader leader-graphene-python.log
    taskset -c 0-7 graphene-sgx python -u leader.py --local-addr=localhost:50051                                  \
                                                     --peer-addr=localhost:50052                                   \
                                                     --data-path=data/leader                                       \
                                                     --checkpoint-path=model/leader/checkpoint                     \
                                                     --export-path=model/leader/saved_model                        \
                                                     --save-checkpoint-steps=10                                    \
                                                     --epoch-num=2                                                 \
                                                     --loglevel=debug 2>&1 | logfilter | tee -a leader-graphene-python.log &
    if [ "$DEBUG" != "0" ]; then
        wait && kill -9 `pgrep -f graphene`
    fi
elif [ "$ROLE" == "follower" ]; then
    rm -rf model/follower follower-graphene-python.log
    taskset -c 8-15 graphene-sgx python -u follower.py --local-addr=localhost:50052                               \
                                                        --peer-addr=localhost:50051                                \
                                                        --data-path=data/follower                                  \
                                                        --checkpoint-path=model/follower/checkpoint                \
                                                        --export-path=model/follower/saved_model                   \
                                                        --save-checkpoint-steps=10                                 \
                                                        --epoch-num=2                                              \
                                                        --loglevel=debug 2>&1 | logfilter | tee -a follower-graphene-python.log &
    if [ "$DEBUG" != "0" ]; then
        wait && kill -9 `pgrep -f graphene`
    fi
fi