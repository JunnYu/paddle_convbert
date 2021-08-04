# 使用PaddlePaddle复现论文：ConvBERT: Improving BERT with Span-based Dynamic Convolution

# install
pip install -e .

# ConvBERT
[ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496)

**摘要：**
像BERT及其变体这样的预训练语言模型最近在各种自然语言理解任务中取得了令人印象深刻的表现。然而，BERT严重依赖全局自注意力块，因此需要大量内存占用和计算成本。
虽然它的所有注意力头从全局角度查询整个输入序列以生成注意力图，但我们观察到一些头只需要学习局部依赖，这意味着存在计算冗余。
因此，我们提出了一种新颖的基于跨度的动态卷积来代替这些自注意力头，以直接对局部依赖性进行建模。新的卷积头与其余的自注意力头一起形成了一个新的混合注意力块，在全局和局部上下文学习中都更有效。
我们为 BERT 配备了这种混合注意力设计并构建了一个ConvBERT模型。实验表明，ConvBERT 在各种下游任务中明显优于BERT及其变体，具有更低的训练成本和更少的模型参数。
值得注意的是，ConvBERT-base 模型达到86.4GLUE分数，比ELECTRA-base高0.7，同时使用不到1/4的训练成本。

本项目是 ConvBert 在 Paddle 2.x上的开源实现。

## 快速开始

### 模型精度对齐
运行`python compare.py`，对比huggingface与paddle之间的精度，我们可以发现精度的平均误差在10^-7量级，最大误差在10^-6量级。
```python
    python compare.py
    # huggingface YituTech/conv-bert-small vs paddle convbert-small
    # mean difference: tensor(4.6980e-07)
    # max difference: tensor(2.8610e-06)
    # huggingface YituTech/conv-bert-medium-small vs paddle convbert-medium-small
    # mean difference: tensor(3.4326e-07)
    # max difference: tensor(2.8014e-06)
    # huggingface YituTech/conv-bert-base vs paddle convbert-base
    # mean difference: tensor(4.5306e-07)
    # max difference: tensor(8.1062e-06)
```

## **数据准备**

### Fine-tuning数据
Fine-tuning 使用GLUE数据，这部分Paddle已提供，在执行Fine-tuning 命令时会自动下载并加载

## **模型预训练**

**特别注意**：预训练模型如果想要达到较好的效果，需要训练几乎全量的Book Corpus数据 和 Wikipedia Corpus数据，原始文本接近20G，建议用GPU进行预训练，最好4片GPU以上。如果资源较少，Paddle提供已经预训练好的模型进行Fine-tuning，可以直接跳转到下面：运行Fine-tuning-使用Paddle提供的预训练模型运行 Fine-tuning

### 单机单卡
```shell
export CUDA_VISIBLE_DEVICES="0"
export DATA_DIR=./BookCorpus/

python -u examples/language_model/convbert/run_pretrain.py \
    --model_type convbert \
    --model_name_or_path convbert-medium-small \
    --input_dir $DATA_DIR \
    --output_dir ./pretrain_model/ \
    --train_batch_size 64 \
    --learning_rate 5e-4 \
    --max_seq_length 128 \
    --weight_decay 1e-2 \
    --adam_epsilon 1e-6 \
    --warmup_steps 10000 \
    --num_train_epochs 4 \
    --logging_steps 100 \
    --save_steps 10000 \
    --max_steps -1 \
    --device gpu
```
其中参数释义如下：
- `model_type` 表示模型类型，默认为ConvBERT模型。
- `model_name_or_path` 如果配置1个名字，则表示预训练模型的规模，当前支持的名字为：convbert-small、convbert-medium-small、convbert-base。如果配置1个路径，则表示按照路径中的模型规模进行训练，这时需配置 --init_from_ckpt 参数一起使用，一般用于断点恢复训练场景。
- `input_dir` 表示输入数据的目录，该目录下需要有1个train.data纯英文文本文件，utf-8编码。
- `output_dir` 表示将要保存预训练模型的目录。
- `train_batch_size` 表示 每次迭代**每张卡**上的样本数目。此例子train_batch_size=64 运行时大致需要单卡12G显存，如果实际GPU显存小于12G或大大多于12G，可适当调小/调大此配置。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `weight_decay` 表示每次迭代中参数缩小的比例，该值乘以学习率为真正缩小的比例。
- `adam_epsilon` 表示adam优化器中的epsilon值。
- `warmup_steps` 表示学习率逐渐升高到基础学习率（即上面配置的learning_rate）所需要的迭代数，最早的使用可以参考[这篇论文](https://arxiv.org/pdf/1706.02677.pdf)。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存间隔。
- `max_steps` 如果配置且大于0，表示预训练最多执行的迭代数量；如果不配置或配置小于0，则根据输入数据量、train_batch_size和num_train_epochs来确定预训练迭代数量
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

另外还有一些额外参数不在如上命令中：
- `use_amp` 表示是否开启混合精度(float16)进行训练，默认不开启。如果在命令中加上了--use_amp，则会开启。
- `init_from_ckpt` 表示是否从某个checkpoint继续训练（断点恢复训练），默认不开启。如果在命令中加上了--init_from_ckpt，且 --model_name_or_path 配置的是路径，则会开启从某个checkpoint继续训练。例如下面的命令从第40000步的checkpoint继续训练：


## **Fine-tuning**

### 运行Fine-tuning

#### **使用Paddle提供的预训练模型运行 Fine-tuning**

以 GLUE/QNLI 任务为例，启动 Fine-tuning 的方式如下：
```shell
export CUDA_VISIBLE_DEVICES=0
export TASK_NAME=QNLI

python -u examples/language_model/convbert/run_glue.py \
    --model_type convbert \
    --model_name_or_path convbert-small \
    --task_name $TASK_NAME \
    --max_seq_length 128 \
    --batch_size 256   \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --logging_steps 100 \
    --save_steps 100 \
    --output_dir ./glue/$TASK_NAME/ \
    --device gpu
```
其中参数释义如下：
- `model_type` 指示了模型类型，当前支持BERT、ELECTRA、ERNIE、CONVBERT模型。
- `model_name_or_path` 模型名称或者路径，其中convbert模型当前仅支持convbert-small、convbert-medium-small、convbert-base几种规格。
- `task_name` 表示 Fine-tuning 的任务，当前支持CoLA、SST-2、MRPC、STS-B、QQP、MNLI、QNLI、RTE。
- `max_seq_length` 表示最大句子长度，超过该长度将被截断。
- `batch_size` 表示每次迭代**每张卡**上的样本数目。
- `learning_rate` 表示基础学习率大小，将于learning rate scheduler产生的值相乘作为当前学习率。
- `num_train_epochs` 表示训练轮数。
- `logging_steps` 表示日志打印间隔。
- `save_steps` 表示模型保存及评估间隔。
- `output_dir` 表示模型保存路径。
- `device` 表示使用的设备类型。默认为GPU，可以配置为CPU、GPU、XPU。若希望使用多GPU训练，将其设置为GPU，同时环境变量CUDA_VISIBLE_DEVICES配置要使用的GPU id。

Fine-tuning过程将按照 `logging_steps` 和 `save_steps` 的设置打印如下格式的日志：

```
global step 100/792, epoch: 0, batch: 99, rank_id: 0, loss: 0.333723, lr: 0.0000970547, speed: 3.6162 step/s
eval loss: 0.295912, acc: 0.8623853211009175, eval done total : 0.5295147895812988 s
global step 200/792, epoch: 0, batch: 199, rank_id: 0, loss: 0.243273, lr: 0.0000830295, speed: 3.6822 step/s
eval loss: 0.249330, acc: 0.8899082568807339, eval done total : 0.508596658706665 s
global step 300/792, epoch: 1, batch: 35, rank_id: 0, loss: 0.166950, lr: 0.0000690042, speed: 3.7250 step/s
eval loss: 0.307219, acc: 0.8956422018348624, eval done total : 0.5816614627838135 s
global step 400/792, epoch: 1, batch: 135, rank_id: 0, loss: 0.185729, lr: 0.0000549790, speed: 3.6896 step/s
eval loss: 0.201950, acc: 0.9025229357798165, eval done total : 0.5364704132080078 s
global step 500/792, epoch: 1, batch: 235, rank_id: 0, loss: 0.132817, lr: 0.0000409537, speed: 3.7708 step/s
eval loss: 0.239518, acc: 0.9094036697247706, eval done total : 0.5128316879272461 s
global step 600/792, epoch: 2, batch: 71, rank_id: 0, loss: 0.163107, lr: 0.0000269285, speed: 3.7303 step/s
eval loss: 0.199408, acc: 0.9139908256880734, eval done total : 0.5226929187774658 s
global step 700/792, epoch: 2, batch: 171, rank_id: 0, loss: 0.082950, lr: 0.0000129032, speed: 3.7664 step/s
eval loss: 0.236055, acc: 0.9025229357798165, eval done total : 0.5140993595123291 s
global step 792/792, epoch: 2, batch: 263, rank_id: 0, loss: 0.025735, lr: 0.0000000000, speed: 4.1180 step/s
eval loss: 0.226449, acc: 0.9013761467889908, eval done total : 0.5103530883789062 s
```

对于 SQuAD v1.1,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" examples/language_model/convbert/squad/run_squad.py \
    --model_type convbert \
    --model_name_or_path convbert-base \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict
 ```

* `model_type`: 预训练模型的种类。如bert，ernie，roberta等。
* `model_name_or_path`: 预训练模型的具体名称。如bert-base-uncased，bert-large-cased等。或者是模型文件的本地路径。
* `output_dir`: 保存模型checkpoint的路径。
* `do_train`: 是否进行训练。
* `do_predict`: 是否进行预测。

训练结束后模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 81.18259224219489,
  "f1": 88.68817481234801,
  "total": 10570,
  "HasAns_exact": 81.18259224219489,
  "HasAns_f1": 88.68817481234801,
  "HasAns_total": 10570
}
```

对于 SQuAD v2.0,按如下方式启动 Fine-tuning:

```shell
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "0" examples/language_model/convbert/squad/run_squad.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --max_seq_length 384 \
    --batch_size 12 \
    --learning_rate 3e-5 \
    --num_train_epochs 2 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --warmup_proportion 0.1 \
    --weight_decay 0.01 \
    --output_dir ./tmp/squad/ \
    --device gpu \
    --do_train \
    --do_predict \
    --version_2_with_negative
 ```

* `version_2_with_negative`: 使用squad2.0数据集和评价指标的标志。

训练结束后会在模型会自动对结果进行评估，得到类似如下的输出：

```text
{
  "exact": 73.25865408910974,
  "f1": 76.63096554166046,
  "total": 11873,
  "HasAns_exact": 73.22874493927125,
  "HasAns_f1": 79.98303877802545,
  "HasAns_total": 5928,
  "NoAns_exact": 73.28847771236333,
  "NoAns_f1": 73.28847771236333,
  "NoAns_total": 5945,
  "best_exact": 74.31988545439232,
  "best_exact_thresh": -2.5820093154907227,
  "best_f1": 77.20521797731851,
  "best_f1_thresh": -1.559523582458496
}
```