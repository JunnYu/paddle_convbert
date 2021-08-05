import argparse
from functools import partial

import paddle
from paddle.io import DataLoader

from tqdm import tqdm
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Tuple, Pad
from paddlenlp.transformers import ConvBertForSequenceClassification, ConvBertTokenizer


parser = argparse.ArgumentParser()

parser.add_argument(
    "--ckpt_path",
    default=None,
    type=str,
    required=True,)
args = parser.parse_args()
args.batch_size = 32
args.max_seq_length = 128

test_ds = load_dataset('glue', "qnli", splits="test")
id2label = dict(zip([0,1],test_ds.label_list))

model = ConvBertForSequenceClassification.from_pretrained(args.ckpt_path)
model.eval()
tokenizer = ConvBertTokenizer.from_pretrained(args.ckpt_path)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment
): fn(samples)

def convert_example(example,
                    tokenizer,
                    max_seq_length=512):
    example = tokenizer(
        example['sentence1'],
        text_pair=example['sentence2'],
        max_seq_len=max_seq_length)
    return example['input_ids'], example['token_type_ids']

trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=args.max_seq_length)
test_ds = test_ds.map(trans_func, lazy=True)
test_batch_sampler = paddle.io.BatchSampler(
    test_ds, batch_size=args.batch_size, shuffle=False)
test_data_loader = DataLoader(
    dataset=test_ds,
    batch_sampler=test_batch_sampler,
    collate_fn=batchify_fn,
    num_workers=2,
    return_list=True)

outputs = []

progress_bar = tqdm(
    range(len(test_data_loader)),
    desc="Predition Iteration",
)
with paddle.no_grad():
    for batch in test_data_loader:
        input_ids, segment_ids = batch
        logits = model(input_ids, segment_ids)
        pred = paddle.argmax(logits,axis=-1).cpu().tolist()
        outputs.extend(list(map(lambda x:id2label[x],pred)))
        progress_bar.update(1)


import pandas as pd

d = {
    "index":list(range(len(outputs))),
    "prediction":outputs
}

pd.DataFrame(d).to_csv("templates/QNLI.tsv",sep="\t",index=False)



