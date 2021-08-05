# ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
# 请参考 logs/GLUE/task名字/args.json，然后配置参数！
python -m paddle.distributed.launch --gpus "0" run_glue.py --配置参数