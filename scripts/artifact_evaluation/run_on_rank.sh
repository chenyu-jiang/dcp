#!/bin/bash

if [ $# -lt 3 ]; then
  echo "Usage: $0 <rank> <hosts file> <local_script> <in_container>"
  exit 1
fi
# Path to the local script
RANK=$1
HOSTS_FILE=$2
LOCAL_SCRIPT=$3
IN_CONTAINER=${4:-1}  # Default to 1 if not provided

# Path where the script should be copied on the remote machine
REMOTE_PATH="/tmp/script.sh"
DOCKER_PATH="/root/run.sh"

DATE_STR=$(date -u "+%b%d_%H_%M")
# Read the list of remote machines from hosts.txt
mapfile -t HOSTS < $HOSTS_FILE

CONTAINER_NAME="dcp"
DOCKER_CP_CMD="docker cp ${REMOTE_PATH} ${CONTAINER_NAME}:${DOCKER_PATH}"

HOST=${HOSTS[$RANK]}
echo "Deploying script to $HOST..."
# Copy the script to the remote machine
scp "$LOCAL_SCRIPT" "$HOST:$REMOTE_PATH"

if [ $? -eq 0 ] && [ $IN_CONTAINER -eq 1 ]; then
    echo "Copying script on $HOST into container $CONTAINER_NAME..."
    ssh "$HOST" "$DOCKER_CP_CMD" 
    if [ $? -eq 0 ]; then
        echo "Copied script on $HOST into container $CONTAINER_NAME."
    else
        echo "Failed to copy script on $HOST into container $CONTAINER_NAME."
    fi
elif [ $? -eq 0 ]; then
    echo "Failed to copy script to $HOST."
fi
echo "--------------------------------------"

DOCKER_EXEC_CMD="docker exec $CONTAINER_NAME bash $DOCKER_PATH"
HOST_EXEC_CMD="bash $REMOTE_PATH"

HOST=${HOSTS[$RANK]}
NNODES=${#HOSTS[@]}
MASTER_ADDR=${HOSTS[0]}  # Let the first host be the master
if [ $IN_CONTAINER -eq 1 ]; then
    echo "Executing in container $CONTAINER_NAME on $HOST..."
    ssh "$HOST" "$DOCKER_EXEC_CMD $RANK $NNODES $MASTER_ADDR" >./N${RANK}_output.txt 2>&1 &
else
    echo "Executing on host $HOST..."
    ssh "$HOST" "$HOST_EXEC_CMD $RANK $NNODES $MASTER_ADDR" >./N${RANK}_output.txt 2>&1 &
fi

wait
echo "All commands finished."