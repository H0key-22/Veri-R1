
corpus_file=/datadisk/corpus/train/corpus.jsonl # jsonl
save_dir=/datadisk/corpus/train/index
retriever_name=e5 # this is for indexing naming
retriever_model=/datadisk/model/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1 python index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 1024 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding
