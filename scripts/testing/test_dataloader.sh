export DCP_LOG_INSTRS=1
export DCP_DEBUG=DEBUG

cd tests/data && ./single_node_launch.sh 4 test_dataloader.py
