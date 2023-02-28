#/bin/sh
export VIRTUAL_ENV=/home/paolochiste/pytorch-venv/
export PATH=/home/paolochiste/pytorch-venv/bin:$PATH
export CUDNN_CONV_WSCAP_DBG=4096
log_file_path="output-param-search-linear-lr.log"
date > $log_file_path
nohup python -u src/testing.py -c "" > $log_file_path &
echo "Launched script."
