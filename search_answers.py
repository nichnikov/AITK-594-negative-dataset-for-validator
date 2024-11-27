import os, re
import pandas as pd
from src.storage import ElasticClient
from src.config import logger


data_path = "/home/an/data/github/QA-Retriver-training/data"
# data_fn = ["bss_2020.csv", "bss_2021.csv", "bss_2022.csv", "bss_2023.csv", "bss_2024.csv"]
data_fn = ["bss_2021.csv"]
es = ElasticClient()

for dfn in data_fn:
    fn = os.path.join(data_path, dfn)
    data_df = pd.read_csv(fn, sep="\t")
    short_queries_df = data_df[data_df["LenQuery"] <= 20][:20]

    wr_index = "write_support_data"
    search_results = []
    k = 1

    for q, lm_q in short_queries_df[["QueryText", "LemQuery"]].itertuples(index=False, name=None):
        logger.info(str(dfn) + str(k) + "/" + str(short_queries_df.shape[0]))

        query = {"multi_match": {"query": lm_q, "fields": ["LemAnswer", "LemQuery"]}}
        # query = {"match": {"LemAnswer": lm_q}}

        search_result = es.search_query(wr_index, query)

        search_result_ = [d["_source"] for d in search_result["hits"]["hits"][:3]] 

        in_d = {"InQuery": q, "InLemQuery": lm_q}
        # search_results +=  [{**in_d, **d} for d in search_result_]
        search_results +=  [{**in_d, **d, **{"label": 1}} if lm_q == d["LemQuery"]
                else {**in_d, **d, **{"label": 0}} for d in search_result_]
        
        k += 1


    search_result_df = pd.DataFrame(search_results)

    fn = re.sub(".csv", "", dfn)
    search_result_df.to_csv(os.path.join("datasets", fn + "_241127.csv"), sep="\t")
    # search_result_df.to_feather(os.path.join("datasets", fn + ".feather"))
