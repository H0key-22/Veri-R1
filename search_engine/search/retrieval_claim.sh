
# DATA_NAME=claim
# DATASET_PATH="/root/data/"
DATASET_PATH="/datadisk/data"
SPLIT='train'
TOPK=5

INDEX_PATH=/datadisk/corpus/index_v1/
CORPUS_PATH=/datadisk/corpus/wiki_corpus_v1.jsonl
SAVE_NAME=e5_${TOPK}.json
MODEL_PATH=/datadisk/model/e5

# INDEX_PATH=/home/peterjin/rm_retrieval_corpus/index/wiki-21
# CORPUS_PATH=/home/peterjin/rm_retrieval_corpus/corpora/wiki/enwiki-dec2021/text-list-100-sec.jsonl
# SAVE_NAME=e5_${TOPK}_wiki21.json

CUDA_VISIBLE_DEVICES=0 python retrieval_claim.py --retrieval_method e5 \
                    --retrieval_topk $TOPK \
                    --index_path $INDEX_PATH \
                    --corpus_path $CORPUS_PATH \
                    --dataset_path $DATASET_PATH \
                    --data_split $SPLIT \
                    --retrieval_model_path $MODEL_PATH \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512 \
