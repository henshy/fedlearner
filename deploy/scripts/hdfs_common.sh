#!/bin/bash

set -ex

# When recongize HADOOP_HOME, export some also useful environment variables for GFile
if [ ! -z $HADOOP_HOME ]
then
    echo "set hadoop env"
    # This is super import for compitable with hadoop and hadoop_current
    if [ -f "$HADOOP_HOME/conf/hadoop-env.sh" ]
    then
        export HADOOP_CONF_DIR=$HADOOP_HOME/conf
        source "$HADOOP_HOME/conf/hadoop-env.sh" &> /dev/null
    else
        export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
        source "$HADOOP_HOME/etc/hadoop/hadoop-env.sh" &> /dev/null
    fi

    if [[ "$USING_MT_HADOOP" == "True" ]]; then
        echo "mt kerberos auth"
        export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
        python3 /app/deploy/scripts/mt_auth_util.py ${HADOOP_USER_NAME}
        export KRB5CCNAME=/tmp/krb5cc_0
        echo $KRB5CCNAME
    fi
    export LD_LIBRARY_PATH=${HADOOP_HOME}/lib/native:${HADOOP_HOME}/lib/native/nfs:${JAVA_HOME}/jre/lib/amd64/server:${LD_LIBRARY_PATH}
    export CLASSPATH=.:$CLASSPATH:$JAVA_HOME/lib/dt.jar:$JAVA_HOME/lib/tools.jar:$($HADOOP_HOME/bin/hadoop classpath --glob)
else
    echo "skipping hadoop env initialized..."
fi

