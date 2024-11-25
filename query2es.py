import os, re
import pandas as pd
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.config import logger

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def dict_handling(dd: dict) -> dict:
    pattern = re.compile(r"\n|&laquo;|&ndash;|&nbsp;|&raquo;|&quot;|\s+")
    clear_answer = pattern.sub(" ", dd["Answer"])
    clear_query = pattern.sub(" ", dd["QueryText"])
    dd["ClearAnswer"] = clear_answer
    dd["ClearQuery"] = clear_query
    dd["AnswerUrls"] = re.findall("(?P<url>https?://[^\s]+)", clear_answer)
    return dd


if __name__ == "__main__":
    es = ElasticClient()
    tknz = TextsTokenizer()
    wr_index = "write_support_data"
    es.delete_index(wr_index)
    es.create_index(wr_index)
    
    chunk_size = 10000
    fale_names = ["2020.xlsx", "2021.xlsx", "2022.xlsx", "2023.xlsx", "2024.xlsx"]
    # fale_names = ["2024.xlsx"]
    for fn in fale_names:
        year = re.findall(r"\d+", fn)[0]
        logger.info("Start year: {}".format(str(year)))
        df = pd.read_excel(os.path.join(os.getcwd(), "data", fn))
        df.fillna("NO", inplace=True)
        df_dicts = df.to_dict(orient="records")
        data_dicts = [dict_handling(d) for d in df_dicts]
        k = chunk_size
        all_exms = len(data_dicts)
        for data_chunk in chunks(data_dicts, chunk_size):
            logger.info("{} start lematization of {} from {} examples".format(str(year), str(k), str(all_exms)))
            lem_queries = [" ".join(lm_q) for lm_q in tknz([d["ClearQuery"] for d in data_chunk])]
            lem_answers = [" ".join(lm_a) for lm_a in tknz([d["ClearAnswer"] for d in data_chunk])]
            for d, lq, la in zip(data_chunk, lem_queries, lem_answers):
                d["LemQuery"] = lq
                d["LemAnswer"] = la
            logger.info("{} end lematization of {} from {} examples".format(str(year), str(k), str(all_exms)))
            logger.info("{} start data sending of {} from {} examples".format(str(year), str(k), str(all_exms)))
            
            es.add_docs(wr_index, data_chunk)
            logger.info("{} end data sending of {} from {}".format(str(year), str(k), str(all_exms)))
            k += chunk_size

    