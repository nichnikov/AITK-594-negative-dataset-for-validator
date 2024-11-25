import os
import json
import pandas as pd
from src.texts_processing import TextsTokenizer
from src.config import (logger,
                        PROJECT_ROOT_DIR)
from src.classifiers import FastAnswerClassifier
from sentence_transformers import SentenceTransformer
from src.config import parameters
from transformers import T5Tokenizer, T5ForConditionalGeneration


stopwords = []
if parameters.stopwords_files:
    for filename in parameters.stopwords_files:
        root = os.path.join(PROJECT_ROOT_DIR, "data", filename)
        stopwords_df = pd.read_csv(root, sep="\t")
        stopwords += list(stopwords_df["stopwords"])

t5_tokenizer = T5Tokenizer.from_pretrained('ai-forever/ruT5-large')
t5_model = T5ForConditionalGeneration.from_pretrained(os.path.join("data", 'models_bss')).to("cuda")

sbert_model = SentenceTransformer(os.path.join(PROJECT_ROOT_DIR, "data", "all_sys_paraphrase.transformers"))
mystem_path = os.path.join(PROJECT_ROOT_DIR, "data", "mystem")
tokenizer = TextsTokenizer(mystem_path)
tokenizer.add_stopwords(stopwords)
classifier = FastAnswerClassifier(tokenizer, parameters, sbert_model, t5_model, t5_tokenizer)
logger.info("service started...")