import os, re
import pandas as pd
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.config import logger


df = pd.read_csv(os.path.join("data", "queries2024_ltn15_with_answr_10000.csv"), sep="\t")
print(df)
print(df.info())

query_texts = df["QueryText"].tolist()
print(query_texts[:5])

tokenizer = TextsTokenizer()

lem_queries = [" ".join(tkns) for tkns in tokenizer(query_texts)]
print(lem_queries[:5])

es = ElasticClient()
wr_index = "write_support_data"
search_results = []
k = 1
for q, lm_q in zip(query_texts, lem_queries):
    print(k, "/", len(query_texts))
    query = {"multi_match": {"query": lm_q,  
                             "fields": ["LemAnswer", "LemQuery"]}}
    search_result = es.search_query(wr_index, query)

    search_result_ = [d["_source"] for d in search_result["hits"]["hits"][:10]] 

    in_d = {"InQuery": q, "InLemQuery": lm_q}
    search_results +=  [{**in_d, **d, **{"label": 1}} if lm_q == d["LemQuery"]
               else {**in_d, **d, **{"label": 0}} for d in search_result_]
    
    k += 1

search_result_df = pd.DataFrame(search_results)
print(search_result_df)

search_result_df.to_csv(os.path.join("results", "found_answers.csv"), sep="\t")