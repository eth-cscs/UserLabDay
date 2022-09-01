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
parser.add_argument('--hf-model', type=str, default='bert-base-uncased',
                    help='Name of the HuggingFace model')
parser.add_argument('--bert-cache-dir', type=str,
                    default=os.path.join(os.getcwd(), 'cache'),
                    help='Path to the cache dir of BERT')
parser.add_argument('--num-epochs', type=int, default=1,
                    help='number of benchmark iterations')
parser.add_argument('--download-only', action='store_true',
                    help='Download model, tokenizer, etc and exit')
parser.add_argument('--test', action='store_true',
                    help='Test after training')
parser = deepspeed.add_config_arguments(parser)
args = parser.parse_args()

slow_tokenizer = BertTokenizer.from_pretrained(
    args.hf_model,
    cache_dir=os.path.join(args.bert_cache_dir, f'_{args.hf_model}-tokenizer')
)
save_path = os.path.join(args.bert_cache_dir, f'{args.hf_model}-tokenizer')
if not os.path.exists(save_path):
    os.makedirs(save_path)
    slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer(os.path.join(save_path, 'vocab.txt'),
                                   lowercase=True)

model = BertForQuestionAnswering.from_pretrained(
    args.hf_model,
    cache_dir=os.path.join(args.bert_cache_dir, f'{args.hf_model}_qa')
)

hf_dataset = load_dataset(
    'squad',
    cache_dir=os.path.join(args.bert_cache_dir, 'datasets')
)

if args.download_only:
    exit()

model.train()

max_len = 384

hf_dataset.flatten()
processed_dataset = hf_dataset.flatten().map(
    lambda example: dpp.process_squad_item_batched(example, max_len,
                                                   tokenizer),
    remove_columns=hf_dataset.flatten()['train'].column_names,
    batched=True,
    num_proc=1
)

train_set = processed_dataset["train"]
train_set.set_format(type='torch')

parameters = filter(lambda p: p.requires_grad, model.parameters())

model_engine, optimizer, trainloader, __ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=parameters,
    training_data=train_set
)

# training
for epoch in range(args.num_epochs):  # loop over the dataset multiple times
    for i, batch in enumerate(trainloader, 0):
        outputs = model(
            input_ids=batch['input_ids'].to(model_engine.device),
            token_type_ids=batch['token_type_ids'].to(model_engine.device),
            attention_mask=batch['attention_mask'].to(model_engine.device),
            start_positions=batch['start_token_idx'].to(model_engine.device),
            end_positions=batch['end_token_idx'].to(model_engine.device))
        # forward + backward + optimize
        loss = outputs[0]
        model_engine.backward(loss)
        model_engine.step()

rank = torch.distributed.get_rank()
if rank == 0:
    model_filename = f'model_finetuned_deepspeed'
    # save model's state_dict
    torch.save(model.state_dict(), model_filename)
    print('Finished Training')

    if args.test:
        # create the model again since the previous one is on the gpu
        model_cpu = BertForQuestionAnswering.from_pretrained(
            "bert-base-uncased",
            cache_dir=os.path.join(args.bert_cache_dir, 'bert-base-uncased_qa')
        )

        # load the model on cpu
        model_cpu.load_state_dict(
            torch.load(model_filename,
                       map_location=torch.device('cpu'))
        )

        # load the model on gpu
        # model.load_state_dict(torch.load(model_filename))
        # model.eval()

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

        start_sample = 0
        num_test_samples = 10
        for i, eval_batch in enumerate(eval_dataloader):
            if i > start_sample:
                testing.EvalUtility(eval_batch, [squad_example_objects[i]],
                                    model_cpu).results()

            if i > start_sample + num_test_samples:
                break
