# task name ["cola","sst-2","mrpc","sts-b","qqp","mnli", "rte", "qnli"]
#[ ckpt_path如下
# 'cola/best-cola_ft_model_440.pdparams',
# 'sst-2/best-sst-2_ft_model_2800.pdparams',
# 'mrpc/best-mrpc_ft_model_320.pdparams',
# 'sts-b/best-sts-b_ft_model_600.pdparams',
# 'qqp/best-qqp_ft_model_32000.pdparams',
# 'mnli/best-mnli_ft_model_27000.pdparams',
# 'rte/best-rte_ft_model_600.pdparams',
# 'wnli/best-wnli_ft_model_60.pdparams',
# 'qnli/best-qnli_ft_model_6600.pdparams'
# ]

python run_predict.py --task_name cola  --ckpt_path cola/best-cola_ft_model_800.pdparams 