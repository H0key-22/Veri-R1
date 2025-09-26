import json
import os
from typing import List, Dict
from tqdm import tqdm
from retrieval import get_retriever, get_dataset

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Run retrieval on claim dataset")
    parser.add_argument('--retrieval_method', required=True)
    parser.add_argument('--retrieval_topk', type=int, default=10)
    parser.add_argument('--index_path', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--data_split', default='train')
    parser.add_argument('--faiss_gpu', action='store_true')
    parser.add_argument('--retrieval_model_path', required=True)
    parser.add_argument('--retrieval_pooling_method', default='mean')
    parser.add_argument('--retrieval_query_max_length', type=int, default=256)
    parser.add_argument('--retrieval_use_fp16', action='store_true')
    parser.add_argument('--retrieval_batch_size', type=int, default=512)

    args = parser.parse_args()
    # adjust index path for dense
    if args.retrieval_method != 'bm25':
        args.index_path = os.path.join(args.index_path, f"{args.retrieval_method}_Flat.index")
    else:
        args.index_path = os.path.join(args.index_path, 'bm25')

    # Load claims and run retrieval
    samples = get_dataset(args)
    queries = [s['claim'] for s in samples[:8192]]
    retriever = get_retriever(args)

    print('Start Retrieving ...')
    # get results and scores
    results, scores = retriever.batch_search(queries, return_score=True)

    # Extract only IDs for each retrieved document
    results_ids = [[doc.get('id') for doc in docs] for docs in results]

    # Write output containing only IDs and scores
    out = 'retrieval_results_ids.json'
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'results': results_ids, 'scores': scores}, f, ensure_ascii=False, indent=2)
    print(f'Retrieval ID results written to {out}')
