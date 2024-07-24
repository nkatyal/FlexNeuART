#!/usr/bin/env python
#
#  Copyright 2014+ Carnegie Mellon University
#
#  Using some bits from CEDR: https://github.com/Georgetown-IR-Lab/cedr
#  which has MIT, i.e., Apache 2 compatible license.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import os
import time
import sys
import math
import argparse
import numpy as np
import wandb

from transformers.optimization import get_constant_schedule_with_warmup
from typing import List

import flexneuart.config
import flexneuart.io.train_data

from flexneuart.models.utils import add_model_init_basic_args

from flexneuart.models.train import run_model, clean_memory
from flexneuart.models.base import ModelSerializer, MODEL_PARAM_PREF
from flexneuart.models.train.batch_obj import BatchObject
from flexneuart.models.train.batching import TrainSamplerFixedChunkSizeGroupByQuery,\
                                             BatchingTrainFixedChunkSize

from flexneuart.models.train.distr_utils import run_distributed, get_device_name_arr, \
                                                enable_spawn, avg_model_params
from flexneuart.models.train.loss import *
from flexneuart.models.train.amp import get_amp_processors

from flexneuart import sync_out_streams, set_all_seeds
from flexneuart.io.json import read_json, save_json
from flexneuart.io.runs import read_run_dict, write_run_dict
from flexneuart.io.qrels import read_qrels_dict
from flexneuart.eval import METRIC_LIST, get_eval_results

from flexneuart.config import TQDM_FILE

from tqdm import tqdm
from collections import namedtuple

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sentence_transformers import InputExample, LoggingHandler, SentenceTransformer, losses, models, util, evaluation


class MSMarcoDataset(Dataset):
    def __init__(self, queries, docs, train_sampler, num_train_pairs, neg_qty_per_query):
        self.queries = queries
        self.docs = docs
        self.train_sampler = train_sampler
        self.num_train_pairs = num_train_pairs
        self.neg_qty_per_query = neg_qty_per_query
        self.iterator = iter(train_sampler)
        self.current_sample = None
        self.current_index = 0;
    
    def __len__(self):
        return self.num_train_pairs * self.neg_qty_per_query
    
    def __getitem__(self, index):
        # For a given query, we get a positive document and a list of negative documents
        # We can keep track of how many triples have been emitted for the query and move on to the 
        # next query after producing neg_qty_per_query triples
        if self.current_sample is None or \
            self.current_index >= self.neg_qty_per_query:
            # get the next sample from the sampler and reset index
            try:
                self.current_sample = next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.train_sampler)
                self.current_sample = next(self.iterator)
            self.current_index = 0
        

        qid = self.current_sample.qid
        pos_id = self.current_sample.pos_id
        neg_id = self.current_sample.neg_ids[self.current_index]

        self.current_index += 1
        
        query = self.queries[qid]
        pos_doc = self.docs[pos_id]
        neg_doc = self.docs[neg_id]
        return InputExample(texts = [query, pos_doc, neg_doc])




def main_cli():
    parser = argparse.ArgumentParser('model training and validation')

    parser.add_argument('--seed', metavar='random seed', help='random seed',
                        type=int, default=42)

    parser.add_argument('--max_query_len', metavar='max. query length',
                        type=int, default=flexneuart.config.DEFAULT_MAX_QUERY_LEN,
                        help='max. query length')

    parser.add_argument('--max_doc_len', metavar='max. document length',
                        type=int, default=flexneuart.config.DEFAULT_MAX_DOC_LEN,
                        help='max. document length')

    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=str, nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=str, required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=str, required=True)

    parser.add_argument('--valid_run', metavar='validation file', help='validation file',
                        type=str, required=True)

    parser.add_argument('--model_out_dir',
                        metavar='model out dir', help='an output directory for the trained model',
                        required=True)

    parser.add_argument('--epoch_qty', metavar='# of epochs', help='# of epochs',
                        type=int, default=10)

    parser.add_argument('--epoch_repeat_qty',
                        metavar='# of each epoch repetition',
                        help='# of times each epoch is "repeated"',
                        type=int, default=1)
    parser.add_argument('--batch_size', metavar='batch size',
                        type=int, default=32, help='batch size')

    parser.add_argument('--batch_size_val', metavar='val batch size',
                        type=int, default=32, help='validation batch size')

    parser.add_argument('--max_query_val', metavar='max # of val queries',
                        type=int, default=None,
                        help='max # of validation queries')

    parser.add_argument('--neg_qty_per_query', metavar='listwise negatives',
                        help='Number of negatives per query for a listwise losse',
                        type=int, default=2)

    parser.add_argument('--init_lr', metavar='init learn. rate',
                        type=float, default=0.001, help='initial learning rate for BERT-unrelated parameters')

    parser.add_argument('--no_shuffle_train', action='store_true',
                        help='disabling shuffling of training data')

    args = parser.parse_args()

    all_arg_names = vars(args).keys()

    # Prepare data for sentence bert format
    dataset = flexneuart.io.train_data.read_datafiles(args.datafiles)
    qrelf = args.qrels
    qrels = read_qrels_dict(qrelf)
    train_pairs_all = flexneuart.io.train_data.read_pairs_dict(args.train_pairs)
    num_docs = len(dataset[1])
    print('# of train. queries:', len(train_pairs_all), ' in the file')

    valid_run = read_run_dict(args.valid_run)
    max_query_val = args.max_query_val
    query_ids = list(valid_run.keys())
    if max_query_val is not None:
        query_ids = query_ids[0:max_query_val]
        valid_run = {k: valid_run[k] for k in query_ids}

    print('# of eval. queries:', len(query_ids), ' in the file', args.valid_run)

    valid_relevant_docs = {qid: set(valid_run[qid].keys()) for qid in valid_run.keys()}

    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    # Create a dataset to produce (query, positive_text, negative_text) triple.
    train_sampler = TrainSamplerFixedChunkSizeGroupByQuery(train_pairs=train_pairs_all,
                                               neg_qty_per_query=args.neg_qty_per_query,
                                               qrels=qrels,
                                               epoch_repeat_qty=args.epoch_repeat_qty,
                                               do_shuffle=not args.no_shuffle_train)

    train_dataset = MSMarcoDataset(queries=dataset[0], docs=dataset[1], \
                                    train_sampler=train_sampler, num_train_pairs=len(train_pairs_all), \
                                    neg_qty_per_query=args.neg_qty_per_query)
    
    evaluator = evaluation.InformationRetrievalEvaluator(queries=dataset[0], corpus=dataset[1], \
                                                        relevant_docs=valid_relevant_docs, corpus_chunk_size=num_docs, \
                                                        show_progress_bar=True, name='MsMarco Document Ranking', \
                                                        )
    print(f'No. of training triples: {len(train_dataset)}')

    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=model.smart_batching_collate)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    optimizer = AdamW(model.parameters(), lr=args.init_lr, eps=1e-8)
    num_epochs = args.epoch_qty
    
    model.old_fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=num_epochs,
        checkpoint_save_steps=len(train_dataloader),
        optimizer_params={"lr": args.init_lr},
        show_progress_bar=True,
        output_path=args.model_out_dir,
        save_best_model=True,
        evaluator=evaluator
    )

if __name__ == '__main__':
    main_cli()
