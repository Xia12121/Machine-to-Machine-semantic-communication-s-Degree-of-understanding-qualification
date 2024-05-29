from units.delta_combination import delta_combination
from units.word_generate import generate_sentence
from units.sentence_compare import multi_round_compare
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn



def cal_similarity_sentences(row):
    print("输入句子: ",row["sentence"],"\n理解程度: ",row["Delta"],"\n换词数量: ",row["Token Count"])
    sender_words, reveiver_words, receiver_selection, word_similarity = delta_combination(row["sentence"], row["Delta"])
    similarity = multi_round_compare(sender_words, reveiver_words, row["Token Count"])
    print("双边理解相似度: ",similarity)
    print("###################################################################")
    return similarity

Data_paraphed = pd.read_excel("Data\\opt\\Data_opt.xlsx")
Data_paraphed["sim"] = Data_paraphed.apply(cal_similarity_sentences, axis=1)

Data.to_excel("Result\\result_5t_to_20t_OPT.xlsx")