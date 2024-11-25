"""
классификатор KNeighborsClassifier в /home/an/Data/Yandex.Disk/dev/03-jira-tasks/aitk115-support-questions
"""
import re
import torch
from src.data_types import Parameters
from src.storage import ElasticClient
from src.texts_processing import TextsTokenizer
from src.config import (logger, 
                        empty_result)
from sentence_transformers import util
from collections import namedtuple

# https://stackoverflow.com/questions/492519/timeout-on-a-function-call

tmt = float(10)  # timeout

def search_result_rep(search_result: []):
    return [{**d["_source"],
             **{"id": d["_id"]},
             **{"score": d["_score"]}} for d in search_result]


class FastAnswerClassifier:
    """Объект для оперирования MatricesList и TextsStorage"""

    def __init__(self, tokenizer: TextsTokenizer, parameters: Parameters, sbert_model, t5_model, t5_tokenizer):
        self.es = ElasticClient()
        self.tkz = tokenizer
        self.prm = parameters
        self.sbert_model = sbert_model
        self.device = "cuda"
        self.t5_model = t5_model.to(self.device)
        self.t5_tkz = t5_tokenizer

    async def get_answer(self, templateId, pubid):
        answer_query = {"bool": {"must": [{"match_phrase": {"templateId": templateId}}, {"match_phrase": {"pubId": pubid}},]}}
        resp = await self.es.search_by_query(self.prm.answers_index, answer_query)
        if resp["hits"]["hits"]:
            search_result = search_result_rep(resp["hits"]["hits"])
            return {"templateId": search_result[0]["templateId"],
                    "templateText": search_result[0]["templateText"]}
        else:
            logger.info("not found answer with templateId {} and pub_id {}".format(str(templateId), str(pubid)))
            return empty_result
    
    def sbert_ranging(self, lem_query: str, score: float, candidates: []):
        text_emb = self.sbert_model.encode(lem_query)
        ids, ets, lm_ets, answs = zip(*candidates)
        candidate_embs = self.sbert_model.encode(lm_ets)
        scores = util.cos_sim(text_emb, candidate_embs)
        scores_list = [score.item() for score in scores[0]]
        the_best_result = sorted(list(zip(ids, ets, lm_ets, answs, scores_list)),
                                             key=lambda x: x[4], reverse=True)[0]
        logger.info("sbert_ranging the_best_result score = {}".format(str(the_best_result[4])))
        if the_best_result[4] >= score:
            return the_best_result
    
    def t5_validate(self, query: str, answer: str, score: float):
        text = query + " Document: " + answer + " Relevant: "
        input_ids = self.t5_tkz.encode(text,  return_tensors="pt").to(self.device)
        outputs=self.t5_model.generate(input_ids, eos_token_id=self.t5_tkz.eos_token_id, 
                                       max_length=64, early_stopping=True).to(self.device)
        outputs_decode = self.t5_tkz.decode(outputs[0][1:])
        outputs_logits=self.t5_model.generate(input_ids, output_scores=True, return_dict_in_generate=True, 
                                              eos_token_id=self.t5_tkz.eos_token_id, 
                                              max_length=64, early_stopping=True)
        sigmoid_0 = torch.sigmoid(outputs_logits.scores[0][0])
        t5_score = sigmoid_0[2].item()
        val_str = re.sub("</s>", "", outputs_decode)
        logger.info("t5_validate answer is {} with score = {}".format(val_str, str(t5_score)))
        if val_str == "Правда" and t5_score >= score:
            return True
        
    
    async def searching(self, text: str, pubid: int, sbert_score: float, t5_score: float, candidates: int):
        """searching etalon by  incoming text"""
        try:
            tokens = self.tkz([text])
            if tokens[0]:
                tokens_str = " ".join(tokens[0])
                search_query={"bool": {"must": [{"match_phrase": {"ParentPubList": pubid}}, 
                                            {"match": {"LemCluster": tokens_str}}]}}
                search_result = await self.es.search_by_query(self.prm.clusters_index, query=search_query)
                if search_result["hits"]["hits"]:
                    result_dicts = search_result_rep(search_result["hits"]["hits"])
                    results_tuples = [(d["ID"], d["Cluster"], d["LemCluster"], d["ShortAnswerText"]) 
                                      for d in result_dicts[:candidates]]
                    sbert_the_best_result = self.sbert_ranging(tokens_str, sbert_score, results_tuples)
                    if sbert_the_best_result: 
                        if self.t5_validate(tokens_str, sbert_the_best_result[3], t5_score):
                            answer = await self.get_answer(sbert_the_best_result[0], pubid)
                            return answer
                        else:
                            logger.info("mouse doesn't validate answer for input text {}".format(str(text)))
                            return empty_result
                    else:
                        logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                        return empty_result
                else:
                    logger.info("elasticsearch doesn't find any etalons for input text {}".format(str(text)))
                    return empty_result
            else:
                logger.info("tokenizer returned empty value for input text {}".format(str(text)))
                return empty_result
        except Exception:
            logger.exception("Searching problem with text: {}".format(str(text)))
            return empty_result