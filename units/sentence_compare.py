from sentence_similarity import SentenceSimilarity as sm
from word_generate import generate_sentence
from PH_Model import T5ParaphrasePaws, PegasusParaphrase

def multi_round_compare(sender_words, reveiver_words, sampling_count = 1, IT_number = 20):
    # 构建一个句子相似度计算对象，用于计算句子相似度
    sm1 = sm()
    T5 = T5ParaphrasePaws()
    PH = PegasusParaphrase()
    Total_similarity = 0
    for it_round in list(range(IT_number)):
        sender,receiver = generate_sentence(sender_words, reveiver_words, sampling_count)
        sender_ph = T5.paraphrase(sender)[0]
        receiver_ph = PH.paraphrase(receiver)[0]
        Total_similarity += sm1.similarity(sender_ph, receiver_ph)
    return Total_similarity/IT_number