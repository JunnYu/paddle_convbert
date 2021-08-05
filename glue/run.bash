# ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
python -m paddle.distributed.launch --gpus "0" run_glue.py --task_name cola 