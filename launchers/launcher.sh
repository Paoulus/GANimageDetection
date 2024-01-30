#/bin/sh
export VIRTUAL_ENV=/pytorch-resnet-venv/
export PATH=/pytorch-resnet-venv/bin/:$PATH
export CUDNN_CONV_WSCAP_DBG=4096
log_file_path="output-param-search-linear-lr.log"
date > $log_file_path
nohup python -u src/main.py --device=cpu -c "GAN Detection Project/config.json" >> $log_file_path &
echo "Launched script."
