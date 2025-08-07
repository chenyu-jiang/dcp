#!/bin/bash

if [ $# -ne 3 ]; then
  echo "Usage: $0 <hosts file> <local file> <container path>"
  exit 1
fi

HOSTS_FILE=$1
LOCAL_FILE=$2
CONTAINER_PATH=$3

# Path where the script should be copied on the remote machine
REMOTE_PATH="/tmp/tmp_file"

# Read the list of remote machines from hosts.txt
mapfile -t HOSTS < $HOSTS_FILE

CONTAINER_NAME="dcp"
DOCKER_CP_CMD="docker cp ${REMOTE_PATH} ${CONTAINER_NAME}:${CONTAINER_PATH}"

for i in "${!HOSTS[@]}"; do
  HOST=${HOSTS[i]}
  RANK=$i
  echo "Copying file to $HOST..."
  # Copy the file to the remote machine
  scp "$LOCAL_FILE" "$HOST:$REMOTE_PATH"

  if [ $? -eq 0 ]; then
    echo "Copying file on $HOST into container $CONTAINER_NAME..."
    ssh "$HOST" "$DOCKER_CP_CMD" 
    if [ $? -eq 0 ]; then
      echo "Copied file on $HOST into container $CONTAINER_NAME."
    else
      echo "Failed to copy file on $HOST into container $CONTAINER_NAME."
    fi
  elif [ $? -eq 0 ]; then
      echo "Failed to copy file to $HOST."
  fi
  # remove the temporary file on the remote host
  ssh "$HOST" "rm -rf $REMOTE_PATH"
  if [ $? -eq 0 ]; then
    echo "Removed temporary file on $HOST."
  else
    echo "Failed to remove temporary file on $HOST."
  fi
  echo "--------------------------------------"
done

wait
echo "All commands finished."