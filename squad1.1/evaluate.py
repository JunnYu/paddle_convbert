
import json
from functools import partial
import paddle

from paddle.io import DataLoader
from args import parse_args
import paddle.nn as nn
from paddlenlp.data import Pad, Dict
from paddlenlp.transformers.convbert.modeling import ConvBertPretrainedModel
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset

from transformers.models.convbert.tokenization_convbert_fast import ConvBertTokenizerFast

args = parse_args()


class ConvBertForQuestionAnswering(ConvBertPretrainedModel):
    def __init__(self, convbert):
        super(ConvBertForQuestionAnswering, self).__init__()
        self.convbert = convbert 
        self.classifier = nn.Linear(self.convbert.config["hidden_size"], 2)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None):
        sequence_output = self.convbert(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits


def prepare_validation_features_single(example, tokenizer, args):
    tokenized_example = tokenizer(
        example["question"],
        example["context"],
        truncation="only_second",
        max_length=args.max_seq_length,
        stride=args.doc_stride,
        return_offsets_mapping=True,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=True,
    )

    data = {
        "input_ids": tokenized_example["input_ids"],
        "token_type_ids": tokenized_example["token_type_ids"],
        "example_id": example["id"],
        "offset_mapping": [
            (o if tokenized_example["token_type_ids"][k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ],
    }

    return data

dev_ds = load_dataset("squad", splits="dev_v1")

tokenizer = ConvBertTokenizerFast.from_pretrained(args.model_name_or_path)

dev_ds.map(
    partial(prepare_validation_features_single, tokenizer=tokenizer, args=args),
    batched=False,
    lazy=False,
)
dev_batch_sampler = paddle.io.BatchSampler(
    dev_ds, batch_size=args.batch_size, shuffle=False
)

dev_batchify_fn = lambda samples, fn=Dict(
    {
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    }
): fn(samples)

dev_data_loader = DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    collate_fn=dev_batchify_fn,
    num_workers=0,
    return_list=True,
)

model = ConvBertForQuestionAnswering.from_pretrained(args.model_name_or_path)
model.eval()

all_start_logits = []
all_end_logits = []

with paddle.no_grad():
    for batch in dev_data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids)

        for idx in range(start_logits_tensor.shape[0]):
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        dev_data_loader.dataset.data,
        dev_data_loader.dataset,
        (all_start_logits, all_end_logits),
        args.version_2_with_negative,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold,
    )

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open(
        "prediction.json", "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(
        examples=dev_data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json,
    )