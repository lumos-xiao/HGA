# coding: utf-8
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pickle
import glove_utils
import random
import nltk

from scipy.sparse import dok_matrix
from collections import defaultdict

# import mxnet as mx

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import re

import attack_utils
import encode_utils


FLAGS = attack_utils.FLAGS

################################################################
# Config the logger
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

# Output the log to the file
# fh = logging.FileHandler('log/our_%s_%s_%d_%s.log' % (FLAGS.pre, FLAGS.data, FLAGS.sn, FLAGS.sigma))
fh = logging.FileHandler('log/ours_%s.log' % FLAGS.log_name)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)

# Output the log to the screen using StreamHandler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

# Add two handler
logger.addHandler(ch)
logger.addHandler(fh)
# logger.info('this is info message')

logger.info('******************************\n\n\n******************************')
################################################################


VOCAB_SIZE = attack_utils.MAX_VOCAB_SIZE
MAX_ITER_NUM = 20   # The maximum number of iteration.
TOP_N = 4   # The number of the synonyms
PATH = FLAGS.data

nth_split = FLAGS.nth_split

# To be convenient, we use the objects in the `attack_utils`
attack_dict = attack_utils.org_dic
attack_inv_dict = attack_utils.org_inv_dic
attack_encode_dict = attack_utils.enc_dic
# attack_embed_mat = None
attack_dist_mat = dist_mat = np.load(('aux_files/small_dist_counter_%s_%d.npy' %(FLAGS.data, VOCAB_SIZE)))


FITNESS_W = 0.5 # default to 0.5
CROSSOVER_COEFF = 0.5   # default to 0.5
VARIATION_COEFF = 0.01   # default to 0.01
POP_MAXLEN = 60 # set the pop size to 60, which is the same to the paper



def clean_str(string):
    """
    Reuse the function in the `se_model`
    """
    return encode_utils.clean_str(string)


def read_text(path):
    """
    Reuse the function in the `encode_utils`
    """
    return encode_utils.read_text(path)


def pick_most_similar_words(src_word, dist_mat, ret_count=10, threshold=None):
    """
    Reuse the function in the `glove_utils`
    """
    return glove_utils.pick_most_similar_words(src_word, dist_mat, ret_count, threshold)


def CalculateTheDifferenceRatio(x1,x2):
    """
    Calculate the difference of two sentences
    """
    x1=x1.split(" ")
    x2=x2.split(" ")
    a=0
    b=0
    short=(len(x1) if(len(x1)<=len(x2)) else len(x2))
    for i in range(short):
        if(x1[i]!=x2[i]):
            a+=1
        b+=1
    ratio=a/b
    return a, ratio

def JudWordPart(search_word):
    """
       Universal (Coarse) Pos tags has  12 categories
        - NOUN (nouns)
        - VERB (verbs)
        - ADJ (adjectives)
        - ADV (adverbs)
        - PRON (pronouns)
        - DET (determiners and articles)
        - ADP (prepositions and postpositions)
        - NUM (numerals)
        - CONJ (conjunctions)
        - PRT (particles)
        - . (punctuation marks)
        - X (a catch-all for other categories such as abbreviations or foreign words)
       """
    if type(search_word) == str:
        y = [search_word]
    y_token = nltk.pos_tag(y, tagset='default')[0][1]
    
    return y_token


def FindTheDifferenceWord(x1,x2):
    """
    If two words in the same position are different, print them!
    """
    x1=x1.split(" ")
    x2=x2.split(" ")
    short=(len(x1) if(len(x1)<=len(x2)) else len(x2))
    a = 0
    logger.info('The length of two words: %d vs %d.' % (len(x1), len(x2)))
    for i in range(short):
        if(x1[i]!=x2[i]):
            logger.info('%d-th word: %s(%d) vs %s(%d).' % (i, x1[i], attack_encode_dict[x1[i]], x2[i], attack_encode_dict[x2[i]]))


def replaceword(x,position,word):
    """Replace the word in `position`"""
    x=x.split(' ')
    x_new=x
    x_new[position]=word
    x_new=' '.join(x_new)
    return x_new


def FindSynonyms(search_word, M=4):
    """
    Select `M` words of the `search_word`.
    """
    search_word=attack_dict[search_word]
    nearest, nearest_dist=pick_most_similar_words(search_word,attack_dist_mat,8,0.5)
    near=[]
    for word in nearest:
        near.append(attack_inv_dict[word])
    if len(near) >= M:
        near = near[:M]
    return near

def Crossover_Multipoint(pop,old_snetence,original_label,Cross_coefficient=0.5,Multipoint=5):
    for i in range(len(pop)):
        temp = pop[i]
        pop[i] = temp.split(' ')
    if len(pop) <= 2:
        
        return pop
    new_pop=pop.copy()
    for i in range(len(pop)):
        if np.random.randn()<Cross_coefficient:
            j=random.randint(1,len(pop)-1)
            new_pop[i]=[]
            cross_list = random.sample(range(1, len(pop[i]) - 1), Multipoint)
            cross_list.sort()

            str_i = ' '.join(pop[i])
            str_j = ' '.join(pop[j])
            fitness_i = FitnessFunction(str_i,old_snetence,original_label)
            fitness_j = FitnessFunction(str_j,old_snetence,original_label)
            pecent_choice = 0.5+(fitness_j - fitness_i)/(fitness_i+fitness_j)
            
            pop.extend(pop[i][0:cross_list[0]])
            for i in range(len(cross_list) - 1):
                if JudStringPart(pop[i], pop[j], cross_list[i], cross_list[i + 1]):
                    pop.extend(pop[i][cross_list[i]:cross_list[i + 1]])
                else:
                    if random.random() < pecent_choice:
                        pop.extend(pop[i][cross_list[i]:cross_list[i + 1]])
                    else:
                        pop.extend(pop[j][cross_list[i]:cross_list[i + 1]])
            pop.extend(pop[i][cross_list[len(cross_list) - 1]:len(pop[i])])
           
    for i in range(len(new_pop)):
        new_pop[i] = ' '.join(new_pop[i])
    return new_pop

def FindBestReplace(sentence,position,near,original_label,N=2):
    result_sentences=[]
    score_list=[]
    sentence_list=[]
    for word in near:
        new_sentence=replaceword(sentence,position,word)
        new_classification,new_score=attack_utils.calculate_clf_score(new_sentence)
        score_list.append(1-new_score[0][original_label])
        # if target==1:
        #     score_list.append(new_score[0])
        # else:
        #     score_list.append(1-new_score[0])
        sentence_list.append(new_sentence)
    if(len(near)<2):
        result_sentences=sentence_list
    else:
        best_score_list=[]
        for i in range(N):
            best_score_list.append(score_list.index(max(score_list)))
            score_list[score_list.index(max(score_list))] = float('-inf')
        for j in best_score_list:
            result_sentences.append(sentence_list[j])
    return result_sentences

def FindSynonyms_initial(search_word, M=50):
    """
    Select `M` words of the `search_word`.
    for initial
    """
    str_searchWord = [search_word]
    search_word=attack_dict[search_word]
    nearest, nearest_dist=pick_most_similar_words(search_word,attack_dist_mat,8,0.5)
    near=[]
    searchWord_token = nltk.pos_tag(str_searchWord)[0][1]
    for word in nearest:
        find_word = attack_inv_dict[word]
        find_token = nltk.pos_tag(find_word)[0][1]
        if find_token == searchWord_token:
            near.append(attack_inv_dict[word])
    if len(near) >= M:
        near = near[:M]
    return near

def Generate_initial_seed_population(x,M=2):
    seed_population = []
    classification, score = attack_utils.calculate_clf_score(x)
    original_label = classification[0]
    x_list = x.split(' ')
    influence_words_list=[]
    influence_score_list=[]
    for i in range(len(x_list)):
        sear_word=x_list[i]
        
        if len(sear_word) >= M and sear_word in attack_dict and attack_dict[sear_word] > 27:
            new_snetence = x_list[0:i]+x_list[i+1:len(x_list)]
            new_snetence = ' '.join(new_snetence)
            new_classification,new_score = attack_utils.calculate_clf_score(new_snetence)
            if new_classification[0] == original_label:
                influence_score = score[0][original_label] - new_score[0][original_label]
            else:
                """score_complement = 1 - score[0][original_label]
                new_score_complement = 1 - new_score[0][original_label]
               influence_score = (score[0][original_label] - new_score[0][original_label]) + (new_score_complement - score_complement)"""
                influence_score = (score[0][original_label] - new_score[0][original_label]) + (new_score[0][new_classification[0]] - score[0][new_classification[0]])
            influence_score_list.append(influence_score)
            influence_words_list.append(i)
    
    pop_initial_num = 50
    if len(influence_score_list) <= pop_initial_num:
        number = len(influence_score_list)
    else:
        number = pop_initial_num
    for i in range(number):
        max_influence_idx=influence_score_list.index(max(influence_score_list))
        max_influence_word=x_list[influence_words_list[max_influence_idx]]
        influence_score_list[max_influence_idx]=float('-inf')
        near=FindSynonyms_initial(max_influence_word)
        if len(near)==0:
            continue
        result_sentences=FindBestReplace(x,influence_words_list[max_influence_idx],near,original_label,N=10)
        seed_population.extend(result_sentences)
    return seed_population


def calculate_sentence_sim_score(old_sentence,new_snetence):
    words1 = old_sentence.split(' ')
    words2 = new_snetence.split(' ')
    # print(words1)
    words1_dict = {}
    words2_dict = {}
    for word in words1:
        # word = word.strip(",.?!;")
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()
        # print(word)
        if word != '' and word in words1_dict:
            num = words1_dict[word]
            words1_dict[word] = num + 1
        elif word != '':
            words1_dict[word] = 1
        else:
            continue
    for word in words2:
        # word = word.strip(",.?!;")
        word = re.sub('[^a-zA-Z]', '', word)
        word = word.lower()
        if word != '' and word in words2_dict:
            num = words2_dict[word]
            words2_dict[word] = num + 1
        elif word != '':
            words2_dict[word] = 1
        else:
            continue
    # print(words1_dict)
    # print(words2_dict)
    # return True
    dic1 = sorted(words1_dict.items(), key=lambda asd: asd[1], reverse=True)
    dic2 = sorted(words2_dict.items(), key=lambda asd: asd[1], reverse=True)
    # print(dic1)
    # print(dic2)

    
    words_key = []
    for i in range(len(dic1)):
        words_key.append(dic1[i][0])  
    for i in range(len(dic2)):
        if dic2[i][0] in words_key:
            # print 'has_key', dic2[i][0]
            pass
        else:  
            words_key.append(dic2[i][0])
    # print(words_key)
    vect1 = []
    vect2 = []
    for word in words_key:
        if word in words1_dict:
            vect1.append(words1_dict[word])
        else:
            vect1.append(0)
        if word in words2_dict:
            vect2.append(words2_dict[word])
        else:
            vect2.append(0)
    # print(vect1)
    # print(vect2)
    sum = 0
    sq1 = 0
    sq2 = 0
    for i in range(len(vect1)):
        sum += vect1[i] * vect2[i]
        sq1 += pow(vect1[i], 2)
        sq2 += pow(vect2[i], 2)
    try:
        result = round(float(sum) / (math.sqrt(sq1) * math.sqrt(sq2)), 2)
    except ZeroDivisionError:
        result = 0.0
    # print(result)
    return result

def Cross_population_selection(old_pop,new_pop,old_sentence,original_label,pop_max_size=60):
    all_pop =old_pop + new_pop
    pop=select_high_fitness(old_sentence,all_pop,original_label)
    return pop

def JudStringPart(x1,x2,start_idx,end_idx):
    x1=x1[start_idx:end_idx]
    x2=x2[start_idx:end_idx]
    short = (len(x1) if (len(x1) <= len(x2)) else len(x2))
    for i in range(short):
        if (x1[i] != x2[i]):
            return False
    return True

def JudgeAdv(pop, original_label,thresold=0.6):
    """
    Judge if there exists some adversarial samples in the population
    """
    for sentence in pop:
        classification,score=attack_utils.calculate_clf_score(sentence)

        classification, score = classification[0], score[0][classification[0]]

        if classification!=original_label:
            print('There is adversarial samples, be other = %.3f, score = %.3f' % (classification, score))
            return sentence
    return None


def FitnessFunction(new_sentence,old_sentence,original_label,a=0.5):
    
    classification,score=attack_utils.calculate_clf_score(new_sentence)
    score1=0
    score1 = 1 - score[0][original_label]
    
    _,score2=CalculateTheDifferenceRatio(new_sentence,old_sentence)
   
    score2 = 1 - score2
    score3 = attack_utils.calculate_sentence_sim_score(old_sentence,new_sentence)
   
    #all_score = a * score1 + (1 - a)  * score2
    all_score=a*score1+(1-a)/2*score2+(1-a)/2*score3 
    return all_score


def select_high_fitness(old_sentence,pop,original_label,pop_max_size=60):
    all_score_list=[]

    if len(pop)<=pop_max_size:
        return pop
    for new_sentence in pop:
        all_score=FitnessFunction(new_sentence,old_sentence,original_label,a=FITNESS_W)
        all_score_list.append(all_score)
    best_allscore_list=[]
    for i in range(pop_max_size):
        best_allscore_list.append(all_score_list.index(max(all_score_list)))
        all_score_list[all_score_list.index(max(all_score_list))]= float('inf')
    new_pop=[]
    for score_index in best_allscore_list:
        new_pop.append(pop[score_index])
    return new_pop


def Crossover(pop,Cross_coefficient=0.5):
    """Cross Over"""
    for i in range(len(pop)):
        temp=pop[i]
        pop[i]=temp.split(' ')
    if len(pop) <= 2:
        return pop
    new_pop=pop.copy()
    for i in range(len(pop)):
        if np.random.randn()<Cross_coefficient:
            j=random.randint(1,len(pop)-1)
            k=random.randint(0,len(pop[i])-1)
            new_pop[i]=pop[i][0:k]+pop[j][k:len(pop[j])]
    for i in range(len(new_pop)):
        new_pop[i]=' '.join(new_pop[i])
    return new_pop


def  Variation(pop,original_label,Variation_coefficient=0.01,M=2):
    """Variation of the population"""
    for i in range(len(pop)):
        if type(pop[i]) == str:
            temp=pop[i]
            pop[i]=temp.split(' ')
    new_pop=[]
    for sentence in pop:
        if np.random.randn()<Variation_coefficient:
            j=random.randint(0,len(sentence)-1)
            if len(sentence[j])>M and sentence[j] in attack_dict and attack_dict[sentence[j]] > 27:
                near=FindSynonyms(sentence[j])
                sentence_temp=' '.join(sentence)
                result_sentences=FindBestReplace(sentence_temp,j,near,original_label)
                new_pop.extend(result_sentences)
            else:
                sentence=' '.join(sentence)
                new_pop.append(sentence)
        else:
            sentence=' '.join(sentence)
            new_pop.append(sentence)
    return new_pop


def attacksingle(sentence,iterations_num=MAX_ITER_NUM,pop_max_size=60, seq=0):
    """attack single words"""
    classification, score = attack_utils.calculate_clf_score(sentence)
    original_label = classification[0]
    logger.info('Attacked samples, classification = %.3f, score = %.3f' % (original_label, score[0][original_label]))
    # target=1
    # if classification==[1]:
    #     target=0

    seed_population=Generate_initial_seed_population(sentence)
    #seed_population=Generate_seed_population(sentence)
    adv_sentence = JudgeAdv(seed_population, original_label)
    if adv_sentence:
        return adv_sentence,0,0
    else:
        seed_population=select_high_fitness(sentence,seed_population,original_label,pop_max_size=POP_MAXLEN)

    pop=seed_population
    find_it = False
    number_th = 0
    times = 0  
    for i in range(iterations_num):
        logger.info("%d-th iteration"%i)
        old_pop=pop
        start = time.clock()
        adv_sentence=JudgeAdv(pop,original_label)
        if adv_sentence:
            find_it = True
            sentence = adv_sentence
            number_th = i
            if i != 0:
                logger.info("current mean iteration costs: %.2f" % (times / i))
            break
        pop=select_high_fitness(sentence,pop,original_label,pop_max_size=POP_MAXLEN)
        pop=Crossover_Multipoint(pop,sentence,original_label,Cross_coefficient=CROSSOVER_COEFF)
        pop=Variation(pop,original_label,Variation_coefficient=VARIATION_COEFF)
        pop=Cross_population_selection(old_pop,pop,sentence,original_label)
        end = time.clock()
        times += (end - start)
        logger.info('%d-th iteration costs time %.2fs' % (i, end-start))

    # if FLAGS.gen_adv:
    #     print('>>>>>> trans_yahoo/%s/%s/ours/%d_%d.txt saved.' % (FLAGS.tr_type, FLAGS.nn_type, seq+1, original_label))
    #     with open('trans_yahoo/%s/%s/ours/%d_%d.txt' % (FLAGS.tr_type, FLAGS.nn_type, seq+1, original_label), 'wt') as fo:
    #         fo.write(sentence)

    if find_it:
        return sentence,number_th,times

    return None,20,times


def sample(sentences, labels):
    sentences = np.array(sentences)
    labels = np.array(labels)

    np.random.seed(0)
    shuffled_idx = np.array([i for i in range(len(sentences))])
    np.random.shuffle(shuffled_idx)

    return list(sentences[shuffled_idx]), list(labels[shuffled_idx]), shuffled_idx


TEST_SIZE = 1000
def attack_main():
    x_sentences, y_label = read_text('%s/test' % FLAGS.data)
    x_sentences, y_label, sampled_idx = sample(x_sentences, y_label)
    print('sampled indexes(top 30): ')
    print(sampled_idx[:30])
    print('-------------------------')

    all_sentences_nums=0
    successful_attack_nums=0
    change_ratio=0
    all_number_th = 0  
    all_times = 0  

    for i, (idx, sentence) in enumerate(zip(sampled_idx, x_sentences)):
        sentence = str(sentence)
        x_len = len(sentence.split())
        # if idx not in common_set:
        #     continue
        if x_len >= 100 or x_len <= 10:
            continue
        classification, score = attack_utils.calculate_clf_score(sentence)
        if y_label[i] != classification[0]:
            print('Error.................. for %d' % idx)
            continue
        logger.info("%d/%d attacking..." % (all_sentences_nums, i+1))
        all_sentences_nums+=1
        adv_sentence,number_th,curr_time=attacksingle(sentence, iterations_num=20, pop_max_size=POP_MAXLEN, seq=i+1)
        if adv_sentence:
            curr_change_num, curr_change_ratio, new_adv_sentence = attack_utils.get_show_diff(sentence, adv_sentence)
            if curr_change_ratio < 0.25:
                successful_attack_nums+=1
                all_times += curr_time

                logger.info("original sentence: %s " % sentence)
                logger.info("adversarial sentence: %s" % new_adv_sentence)
                if FLAGS.pre=='org' and FLAGS.gen_adv:
                    classification_adv, score_adv = attack_utils.calculate_clf_score(new_adv_sentence)
                    attack_utils.save_adv_samples_1(new_adv_sentence, 'HGA_adv_samples/%s/%s/%d_%d_%d.txt' % (FLAGS.data, FLAGS.nn_type, idx, y_label[i], classification_adv[0]))
                change_ratio += curr_change_ratio
                all_number_th += number_th

                logger.info("iteration count: %d" % number_th)
                logger.info("current create sample cost time: %.2f" % curr_time)

                logger.info("Current change number: %d, change ratio: %.3f" % (curr_change_num, curr_change_ratio))
                logger.info("Current mean change ratio: %.3f" % (change_ratio/successful_attack_nums))
        logger.info("Current attack success rate: %.3f" % (successful_attack_nums/all_sentences_nums))
        if successful_attack_nums >= TEST_SIZE:
            break
    successful_attack_ratio=successful_attack_nums/all_sentences_nums
    logger.info("Total cost times: %.2f" % all_times)
    logger.info("Total mean iteration cost time: %.2f" % (all_times / successful_attack_nums))
    logger.info("Total iteration count: %d" % all_number_th)
    logger.info("Total attack success rate: %.3f" % successful_attack_ratio)
    logger.info("Total mean change ratio: %.3f " % (change_ratio/successful_attack_nums))


attack_main()
