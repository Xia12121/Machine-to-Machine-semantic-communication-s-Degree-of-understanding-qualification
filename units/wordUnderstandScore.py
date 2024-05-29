import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn

class UnderstandDegree:
    def __init__(self):
        # 下载所需的nltk数据
        nltk.download('wordnet', quiet=True)

    def calcMatch(self, synsets1_dict, synsets2_dict):
        # 比较两个字典中对应词的词义，并计算匹配值
        match_scores = []
        for word, synset1 in synsets1_dict.items():
            if word in synsets2_dict:
                synset2 = synsets2_dict[word]
                match_score = 1 if synset1 == synset2 else 0
                match_scores.append(match_score)
        return match_scores

    def calcImportance(self, sentence_dict):
        '''
        # 计算每个实词的权重，使得所有实词的权重之和为1
        num_filtered_words = len(sentence_dict)
        if num_filtered_words == 0:
            return []
        weight_per_word = 1 / num_filtered_words
        # 为每个实词分配相同的权重
        importance_scores = [weight_per_word] * num_filtered_words
        '''
        importance_scores = 1
        return importance_scores

    def calcDifficulty(self, sentence_dict):
        # 计算每个筛选出的词的词义数量
        num_meanings_list = [len(wn.synsets(word)) for word in sentence_dict.keys()]
        # 计算所有筛选出的词的词义数量之和
        total_meanings = sum(num_meanings_list)
        # 计算每个筛选出的词的难度分数
        difficulty_scores = [num_meanings / total_meanings for num_meanings in num_meanings_list]
        return difficulty_scores

    def calcScores(self, match_scores, difficulty_scores):
        # 计算理解程度分数
        calc_scores = [m * d for m, d in zip(match_scores, difficulty_scores)]
        sim_score = sum(calc_scores)
        return sim_score

    def determineLevel(self, sim_score, thresholds, levels):
        # 确定理解程度等级
        for threshold, level in zip(thresholds, levels):
            if sim_score <= threshold:
                return level
        return levels[-1]

    def simScores(self, synsets1_dict, synsets2_dict):
        # 计算两个句子的相似度分数
        match_scores = self.calcMatch(synsets1_dict, synsets2_dict)
        difficulty_scores = self.calcDifficulty(synsets1_dict)
        sim_score = self.calcScores(match_scores, difficulty_scores)
        return sim_score