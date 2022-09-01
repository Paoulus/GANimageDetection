#/bin/sh
export VIRTUAL_ENV=/home/paolochiste/pytorch-venv/
export PATH=/home/paolochiste/pytorch-venv/bin:$PATH
export CUDNN_CONV_WSCAP_DBG=4096
nohup python -u src/main.py --device=cuda:0 > output-finetuning-full-run.log &
echo "Launched script. Output will be shown in $PWD/output.log"