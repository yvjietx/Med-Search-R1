file_path=/home/yvjie/Search-R1/data/retrieval-corpus
index_file=$file_path/bm25
corpus_file=$file_path/wiki-18.jsonl
retriever_name=bm25

python search_r1/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name
                                            
