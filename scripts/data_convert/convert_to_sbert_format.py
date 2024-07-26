import argparse
import json

import flexneuart.io.train_data
from flexneuart.io.json import read_json, save_json
from flexneuart.io.qrels import read_qrels_dict

from flexneuart.models.train.batching import TrainSamplerFixedChunkSizeUnique

from tqdm import tqdm

def main_cli():
    parser = argparse.ArgumentParser('Convert CEDR format to sentence bert triplets')
    
    parser.add_argument('--datafiles', metavar='data files', help='data files: docs & queries',
                        type=str, nargs='+', required=True)

    parser.add_argument('--qrels', metavar='QREL file', help='QREL file',
                        type=str, required=True)

    parser.add_argument('--train_pairs', metavar='paired train data', help='paired train data',
                        type=str, required=True)
    
    parser.add_argument('--output_dir', metavar='path to store the file containing training triplets', help='output path',
                        type=str, required=True)
    
    parser.add_argument('--neg_qty_per_query', metavar='listwise negatives',
                        help='Number of negatives per query for a listwise loss',
                        type=int, default=2)

    args = parser.parse_args()

    queries, docs = flexneuart.io.train_data.read_datafiles(args.datafiles)

    qrelf = args.qrels
    qrels = read_qrels_dict(qrelf)

    train_pairs_all = flexneuart.io.train_data.read_pairs_dict(args.train_pairs)
    
    expected_num_train_pairs = 0
    for query_id in train_pairs_all.keys():
        expected_num_train_pairs += len(train_pairs_all[query_id].keys())

    train_sampler = TrainSamplerFixedChunkSizeUnique(train_pairs=train_pairs_all,
                                               neg_qty_per_query=args.neg_qty_per_query,
                                               qrels=qrels,
                                               epoch_repeat_qty=1,
                                               do_shuffle=False)

    iterator = iter(train_sampler)

    train_triplets = []
    while True:
        try:
            group = next(iterator)

            # for each query club the positive doc id with a negative doc id
            qid = group.qid
            pos_id = group.pos_id
            neg_ids = group.neg_ids

            for neg_id in neg_ids:
                train_triplets.append((qid, pos_id, neg_id))
        except StopIteration:
            break

    triplet_id_file_path = args.output_dir + "/train_triplets_id.jsonl"
    with open(triplet_id_file_path, 'w') as triplet_ids_file:
        for qid, pos_id, neg_id in train_triplets:
            d = {'qid': qid, 'positive_docid': pos_id, 'negative_doc_id': neg_id}
            triplet_ids_file.write(json.dumps(d) + '\n')
    
    triplet_file_path = args.output_dir + "/train_triplets.jsonl"
    with open(triplet_file_path, 'w') as triplet_file:
        for qid, pos_id, neg_id in train_triplets:
            d = {'query': queries[qid], 'positive_document': docs[pos_id], 'negative_document': docs[neg_id]}
            triplet_file.write(json.dumps(d) + '\n')

    print('Successfully wrote triplet ids file at ', triplet_id_file_path)
    print('Successfully wrote triplet file at ', triplet_file_path)
if __name__ == '__main__':
    main_cli()