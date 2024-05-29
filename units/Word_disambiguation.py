from nltk.corpus import wordnet as wn
from nltk import pos_tag
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_similarity import SentenceSimilarity as sim
import itertools
# sentence_similarity 是通过hugging face下载的模型 通过这个模型计算两个句子之间的相似度

def tokenizer(sentence):
    # 分词器，通过分词器将句子拆分成一个个单词
    # 输入：完整的句子
    # 输出：一个由句子中所有单词组成的list
    return nltk.word_tokenize(sentence)

def location_target_word(sentence):
    original_word = []
    target_list = tokenizer(sentence)
    index = []
    flag = 0
    for word in target_list:
        synsets = wn.synsets(word)
        if len(synsets)>=2:
            index.append(flag)
            original_word.append(word)
        flag += 1
    return original_word,index

lemmatizer = WordNetLemmatizer()

def get_base_form(word, pos):
    """获取单词的基本形式，以更好地比较不同形态的词"""
    return lemmatizer.lemmatize(word, pos=pos)

def nltk_pos_to_wordnet_pos(nltk_pos):
    """将NLTK的词性标签转换为WordNet的词性"""
    if nltk_pos.startswith('J'):
        return wn.ADJ
    elif nltk_pos.startswith('V'):
        return wn.VERB
    elif nltk_pos.startswith('N'):
        return wn.NOUN
    elif nltk_pos.startswith('R'):
        return wn.ADV
    else:       
        return None

def extract_synsets(word):
    '''提取对应单词所有可能的语义并且排除同型词的干扰'''
    synsets = wn.synsets(word)
    replacements = []

    nltk_pos = pos_tag([word])[0][1]  # 使用NLTK的词性标注
    wordnet_pos = nltk_pos_to_wordnet_pos(nltk_pos)  # 转换为WordNet的词性
    base_form = get_base_form(word, wordnet_pos) if wordnet_pos else word

    for synset in synsets:
        lemmas = synset.lemma_names()
        replacement_found = False

        for lemma in lemmas:
            lemma_base_form = get_base_form(lemma, wordnet_pos) if wordnet_pos else lemma
            if lemma_base_form != base_form:
                replacements.append((synset, lemma))
                replacement_found = True
                break
        
        if not replacement_found:
            # 尝试从上位词中找替换词
            for hypernym in synset.hypernyms():
                for lemma in hypernym.lemma_names():
                    lemma_base_form = get_base_form(lemma, wordnet_pos) if wordnet_pos else lemma
                    if lemma_base_form != base_form:
                        replacements.append((hypernym, lemma))
                        replacement_found = True
                        break
                if replacement_found:
                    break

        if not replacement_found:
            # 如果上位词也找不到，则尝试从下位词中找替换词
            for hyponym in synset.hyponyms():
                for lemma in hyponym.lemma_names():
                    lemma_base_form = get_base_form(lemma, wordnet_pos) if wordnet_pos else lemma
                    if lemma_base_form != base_form:
                        replacements.append((hyponym, lemma))
                        replacement_found = True
                        break
                if replacement_found:
                    break
        
        if not replacement_found:
            replacements.append((synset, "None"))

    return replacements

def replacement_of_sentence(sentence,original_word,indexs, index):
    Word_list = tokenizer(sentence)
    synsets_list = extract_synsets(Word_list[indexs[index]])
    result = {}
    for synset in synsets_list:
        temp_sentence = Word_list
        temp_sentence[indexs[index]] = synset[1]
        temp_sentence = ' '.join(temp_sentence)
        synset_name = synset[0].name()
        result[synset_name] = temp_sentence
    return result

def disambiguation(original_sentence, sentence_dictionary,target_word):
    '''对于单字的消歧义'''
    similarity_caculator = sim()
    synset_list = list(sentence_dictionary.keys())
    sentence_list = list(sentence_dictionary.values())
    similarity = []
    for sentence in sentence_list:
        similarity.append(similarity_caculator.similarity(original_sentence,sentence))
    max_similarity = max(similarity)
    max_index = similarity.index(max_similarity)
    return target_word,synset_list[max_index],sentence_list[max_index]

def disambiguation_sentence(sentence):
    original_word,indexs = location_target_word(sentence)
    original_sentence = sentence
    flag = 0
    word_result = {}
    for word in original_word:
        sentence_dictionary = replacement_of_sentence(sentence,original_word,indexs, flag)
        flag += 1
        target_word,synset,sentence = disambiguation(sentence, sentence_dictionary,word)
        word_result[target_word] = synset
    return original_sentence,sentence, word_result

def synset_combinations(input_dict):
    # 获取每个单词的所有同义词集名称
    synsets_dict = {word: [synset.name() for synset in wn.synsets(word)] for word in input_dict.keys()}

    # 生成所有单词的同义词集的所有组合
    keys, synset_lists = zip(*synsets_dict.items())
    all_combinations = itertools.product(*synset_lists)

    # 为每种组合创建一个字典
    combinations_dicts = [dict(zip(keys, combination)) for combination in all_combinations]

    return combinations_dicts