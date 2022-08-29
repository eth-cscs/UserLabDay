import argparse
import deepspeed
import os
import utility.data_processing as dpp
import utility.testing as testing
import torch
from datasets import load_dataset, load_metric
from transformers import BertTokenizer, BertForQuestionAnswering, AdamW
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datetime import datetime
from datasets.utils import disable_progress_bar
from datasets import disable_caching


disable_progress_bar()
disable_caching()

parser = argparse.ArgumentParser(description='BERT finetuning on SQuAD')
parser.add_argument('--model-file', type=str,
                    help='Finetuned model file')
parser.add_argument('--start-sample', type=int, default=0,
                    help='ID of the first sample to evaluate')
parser.add_argument('--num-test-samples', type=int, default=10,
                    help='Number of samples to evaluate')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

hf_model = 'bert-base-uncased'
bert_cache = os.path.join(os.getcwd(), 'cache')

slow_tokenizer = BertTokenizer.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'_{hf_model}-tokenizer')
)
save_path = os.path.join(bert_cache, f'{hf_model}-tokenizer')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model_cpu = BertForQuestionAnswering.from_pretrained(
    hf_model,
    cache_dir=os.path.join(bert_cache, f'{hf_model}_qa')
)

hf_dataset = load_dataset('squad')

max_len = 384

hf_dataset.flatten()
processed_dataset = hf_dataset.flatten().map(
    lambda example: dpp.process_squad_item_batched(example, max_len,
                                                   tokenizer),
    remove_columns=hf_dataset.flatten()['train'].column_names,
    batched=True,
    num_proc=12
)

# load the model on cpu
model_cpu.load_state_dict(
    torch.load(args.model_file,
               map_location=torch.device('cpu'))
)

eval_set = processed_dataset["validation"]
eval_set.set_format(type='torch')
batch_size = 1

eval_dataloader = DataLoader(
    eval_set,
    shuffle=False,
    batch_size=batch_size
)

squad_example_objects = []
for item in hf_dataset['validation'].flatten():
    squad_examples = dpp.squad_examples_from_dataset(item, max_len,
                                                     tokenizer)
    try:
        squad_example_objects.extend(squad_examples)
    except TypeError:
        squad_example_objects.append(squad_examples)

assert len(eval_set) == len(squad_example_objects)

for i, eval_batch in enumerate(eval_dataloader):
    if i > args.start_sample:
        testing.EvalUtility(eval_batch, [squad_example_objects[i]],
                            model_cpu).results()

    if i > args.start_sample + args.num_test_samples:
        break
