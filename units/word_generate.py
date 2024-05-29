import random
from collections import OrderedDict
from nltk.corpus import wordnet as wn

def replace_keys_with_synonyms(sender_dict, receiver_dict, num_replacements=1):
    # 确保输入的数字不大于字典中的键数
    num_replacements = min(num_replacements, len(receiver_dict))
    
    # 随机选择指定数量的键
    keys_to_replace = random.sample(list(receiver_dict.keys()), num_replacements)
    
    # 使用OrderedDict保持键的顺序
    updated_dict = OrderedDict()
    
    for key in receiver_dict:
        if key in keys_to_replace:
            synset_name = receiver_dict[key]
            synset = wn.synset(synset_name)
            new_key = None
            
            # 查找不同的同义词
            for lemma in synset.lemmas():
                if lemma.name().replace('_', ' ') != key:
                    new_key = lemma.name().replace('_', ' ')
                    break
            
            # 如果没有找到，尝试上位词的同义词
            if not new_key:
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        new_key = lemma.name().replace('_', ' ')
                        break
                    if new_key:
                        break
            
            # 如果还没有找到，尝试下位词的同义词
            if not new_key:
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        new_key = lemma.name().replace('_', ' ')
                        break
                    if new_key:
                        break
            
            # 如果找到了新键，更新字典
            if new_key:
                updated_dict[new_key] = receiver_dict[key]
            else:
                updated_dict[key] = receiver_dict[key]
        else:
            updated_dict[key] = receiver_dict[key]
    
    # 使用OrderedDict保持键的顺序
    '''对于sender的修改'''
    sender = OrderedDict()
    
    for key in sender_dict:
        if key in keys_to_replace:
            synset_name = sender_dict[key]
            synset = wn.synset(synset_name)
            new_key = None
            
            # 查找不同的同义词
            for lemma in synset.lemmas():
                if lemma.name().replace('_', ' ') != key:
                    new_key = lemma.name().replace('_', ' ')
                    break
            
            # 如果没有找到，尝试上位词的同义词
            if not new_key:
                for hypernym in synset.hypernyms():
                    for lemma in hypernym.lemmas():
                        new_key = lemma.name().replace('_', ' ')
                        break
                    if new_key:
                        break
            
            # 如果还没有找到，尝试下位词的同义词
            if not new_key:
                for hyponym in synset.hyponyms():
                    for lemma in hyponym.lemmas():
                        new_key = lemma.name().replace('_', ' ')
                        break
                    if new_key:
                        break
            
            # 如果找到了新键，更新字典
            if new_key:
                sender[new_key] = sender_dict[key]
            else:
                sender[key] = sender_dict[key]
        else:
            sender[key] = sender_dict[key]
    
    
    return dict(sender), dict(updated_dict)

def generate_sentence(senders_words, reveiver_words, change_count=1):
    receiver_words_list = list(replace_keys_with_synonyms(senders_words,reveiver_words, change_count)[1].keys())
    sender_word_list = list(replace_keys_with_synonyms(senders_words,reveiver_words, change_count)[0].keys())
    Receiver_sentence = ' '.join(receiver_words_list)
    Sender_sentence = ' '.join(sender_word_list)
    return Sender_sentence, Receiver_sentence