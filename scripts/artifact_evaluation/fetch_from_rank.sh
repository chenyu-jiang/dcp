#!/bin/bash

if [ $# -le 4 ]; then
  echo "Usage: $0 <rank> <hosts file> <local_path> <remote_container_path>"
  exit 1
fi
RANK=$1
HOSTS_FILE=$2
LOCAL_PATH=$3
DOCKER_PATH=$4


# Path where the script should be copied on the remote machine
REMOTE_PATH="/tmp/tmp_obj"

# Read the list of remote machines from hosts.txt
mapfile -t HOSTS < $HOSTS_FILE

CONTAINER_NAME="dcp"
DOCKER_CP_CMD="docker cp ${CONTAINER_NAME}:${DOCKER_PATH} ${REMOTE_PATH}"

HOST=${HOSTS[$RANK]}
echo "Copying object from container to host..."
ssh "$HOST" "$DOCKER_CP_CMD" 
if [ $? -eq 0 ]; then
    echo "Copied object from container to host."
else
    echo "Failed to copy object from container to host."
    exit 1
fi
echo "Copying object from host to local path..."
scp -r "$HOST:$REMOTE_PATH" "$LOCAL_PATH"
if [ $? -eq 0 ]; then
    echo "Copied object from host to local path."
else
    echo "Failed to copy object from host to local path."
    exit 1
fi
echo "--------------------------------------"
echo "All commands finished."