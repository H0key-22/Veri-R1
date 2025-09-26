
# file_path=/the/path/you/save/corpus
index_file=/datadisk/corpus/nq/e5_Flat.index
corpus_file=/datadisk/corpus/nq/wiki-18.jsonl
retriever_name=e5
retriever_path=/datadisk/model/e5-base-v2

python search_engine/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu
