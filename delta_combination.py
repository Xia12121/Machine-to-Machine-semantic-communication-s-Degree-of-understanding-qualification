from Word_disambiguation import disambiguation_sentence, synset_combinations
from wordUnderstandScore import UnderstandDegree
import math
import random
from nltk.corpus import wordnet

def difficulty_of_word(sender_word):
    total_num = 0
    for word in sender_word:
        total_num += len(wordnet.synsets(word))
    word_difficulty = {}
    for word in sender_word:
        word_difficulty[word] = len(wordnet.synsets(word))/total_num
    return word_difficulty

'''背包问题代码 不要改动这个是万年代码'''
#####################################
def knapsack(weights, target_sum):
    n = len(weights)  # 获取物品数量
    # 初始化dp数组
    dp = [[0 for _ in range(target_sum + 1)] for _ in range(n + 1)]

    # 填充dp数组
    for i in range(1, n + 1):
        for j in range(1, target_sum + 1):
            if j < weights[i-1]:  # 当前物品重量超过当前背包容量
                dp[i][j] = dp[i-1][j]
            else:
                # 选择当前物品或不选择当前物品的最优解
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + weights[i-1])

    # 回溯找到选中的物品
    res = dp[n][target_sum]
    selected = [0] * n
    for i in range(n, 0, -1):
        if res <= 0:
            break
        if dp[i][res] == dp[i-1][res]:
            continue
        else:
            selected[i-1] = 1
            res -= weights[i-1]

    return selected
#####################################

def word_synsets_selection(dictionary, target_delta):
    weights = [int(val * 1000000) for val in dictionary.values()]
    target_weight_sum_int = int(target_delta * 1000000)
    # 避免计算机浮点数操作
    selected_items = knapsack(weights, target_weight_sum_int)
    result_dict = {word: selected for word, selected in zip(dictionary.keys(), selected_items)}
    return result_dict

from nltk.corpus import wordnet as wn

def replace_synsets(dict1, dict2):
    result = {}
    for key, value in dict2.items():
        if value == 1:  # 如果值为1，保留原来的synset
            result[key] = dict1[key]
        else:  # 如果值为0，查找并替换synset
            original_synset = wn.synset(dict1[key])
            # 获取所有synsets，排除当前的synset
            synsets = [s for s in wn.synsets(key) if s != original_synset]
            if synsets:
                # 随机选择一个synset作为与原来synset不同的语义选择
                result[key] = random.choice(synsets).name()
            else:
                # 如果没有其他synsets，保留原来的
                result[key] = dict1[key]
    return result
    

def delta_combination(sentence, delta):
    sender_words = disambiguation_sentence(sentence)[2]
    # 通过sender的消歧实现对sender对语义的掌握
    words_weights = difficulty_of_word(sender_words)
    receiver_selection = word_synsets_selection(words_weights, delta)
    reveiver_words = replace_synsets(sender_words, receiver_selection)
    
    # 通过对所有可能语义的排列组合实现对语义的尝试
    ud = UnderstandDegree()
    word_similarity = ud.simScores(sender_words,reveiver_words)
    return sender_words, reveiver_words, receiver_selection, word_similarity