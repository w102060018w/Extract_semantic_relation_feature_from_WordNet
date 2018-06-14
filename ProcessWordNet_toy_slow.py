'''
# coding: utf-8

# In[4]:


import numpy as np
import nltk
from nltk.corpus import wordnet as wn
relation_feat_syn = None
relation_feat_ant = None
relation_feat_hype = None
relation_feat_hypo = None
relation_feat_same_hype = None
def PrintRelationLst():
    print('relation-syn:',relation_feat_syn)
    print('relation-ant:',relation_feat_ant)
    print('relation-hype:',relation_feat_hype)
    print('relation-hypo:',relation_feat_hypo)
    print('relation-same_hype:',relation_feat_same_hype)


# ## Synonymy & Antonymy

# In[2]:


# WORD1 = 'good'
# WORD2 = 'beneficial'
# WORD1 = 'wet'
# WORD2 = 'dry'
WORD1 = 'wet'
WORD2 = 'dry'
syn_flg = False
WORD1_syn_lst = wn.synsets(WORD1)
WORD2_syn_lst =wn.synsets(WORD2)
# synonymy
for word1_i, word1_syn in enumerate(WORD1_syn_lst):
    for word2_i, word2_syn in enumerate(WORD2_syn_lst):
        if word1_syn == word2_syn:
            relation_feat_syn = 1
    
# antonymy
syn_flg = True
for word1_i, word1_syn in enumerate(WORD1_syn_lst):
    for word2_i, word2_syn in enumerate(WORD2_syn_lst):
        if word1_syn.lemmas()[0].antonyms():
#             print('word1:',word1_syn.lemmas()[0])
#             print('word2:',word2_syn.lemmas()[0])
#             print('============================')
            if word1_syn.lemmas()[0].antonyms()[0] == word2_syn.lemmas()[0]:
                relation_feat_ant = 1
if not relation_feat_syn:
    relation_feat_syn = 0
if not relation_feat_ant:
    relation_feat_ant = 0
PrintRelationLst()


# ## Hypernymy & Hpyponymy & Same-Hypernym

# In[3]:


# here due to its external knowledge definition, we will simple consider Hypernymy and Hyponymy as the same value
# =========== def ===========
# takes value 1 - n/8 if one word is a (direct or indirect) hypernym of the other wors in WordNet. 
# Where n is the number of edges between the two words in hierarchies, and 0 otherwise.
# Eg. [dog, canid] = 0.875
# Eg. [wolf, canid] = 0.875
# Eg. [dog, carnivore] = 0.75
# Eg. [canid, dog] = 0
# ======== End def ==========

## first find their most similarity def
print('word1 = ',WORD1)
print('word2 = ',WORD2)
print('================')
similarity = -np.float('inf')
for word1_i, word1_syn in enumerate(WORD1_syn_lst):
    for word2_i, word2_syn in enumerate(WORD2_syn_lst):
        sim_val = wn.wup_similarity(word1_syn, word2_syn)
        if sim_val: # if they exist similaroty value
            if sim_val > similarity:
                best_val = sim_val
                best_syn1 = word1_syn
                best_syn2 = word2_syn
                similarity = sim_val
# print(best_syn1)
# print(best_syn2)
        
## base on the def, to figure out their hyper-path
best_syn1_path = best_syn1.hypernym_paths()
for idx, path in enumerate(best_syn1_path):
    for i, hyper in enumerate(path):
        if hyper == best_syn2:
            edgN = len(path)-i-1
            relation_feat_hype = 1-edgN/8
            break
    if relation_feat_hype:
        break
    
best_syn2_path = best_syn2.hypernym_paths()
for idx, path in enumerate(best_syn2_path):
    for i, hyper in enumerate(path):
        if hyper == best_syn1:
            edgN = len(path)-i-1
            relation_feat_hypo = 1-edgN/8
            break
    if relation_feat_hypo:
        break

if relation_feat_hype:
    relation_feat_hypo = relation_feat_hype
elif relation_feat_hypo:
    relation_feat_hype = relation_feat_hypo
else:
    relation_feat_hype = 0
    relation_feat_hypo = 0
    
## Same-Hypernym
if relation_feat_syn == 0:
    # to see if the two words have the same hypernym
    for syn1_idx, syn1_path in enumerate(best_syn1_path):
        for syn2_idx, syn2_path in enumerate(best_syn2_path):
            for syn1_hyper in syn1_path:
                for syn2_hyper in syn2_path:
                    if syn1_hyper == syn2_hyper:
#                         print(syn1_hyper)
#                         print(syn2_hyper)
                        relation_feat_same_hype = 1
if not relation_feat_same_hype:
    relation_feat_same_hype = 0
# PrintRelationLst() 


# In[4]:


PrintRelationLst()


# In[5]:


## External knowledge

#rij = [syn, ant, hype, hypo, same-hype]
#λ1(rij ), where λ is a hyper-parameter tuned on the development set 
#and 1 is the indication function.
#1(rij ) = 1 if rij is not zero vector ;
#1(rij ) = 0 if rij is zero vector .

r = [relation_feat_syn, relation_feat_ant, relation_feat_hype, relation_feat_hypo, relation_feat_same_hype] # r_ij
if all(ele == 0 for ele in r):
    lr = 0 # lr_ij
else:
    lr = 1 # lr_ij
'''

# ## from SNLI-dataset to build a relation-feature dictionary

# In[5]:


## load data
import json
import random    
import string
import numpy as np
import nltk
from nltk.corpus import wordnet as wn

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": 0
}

def load_nli_data(path, snli=False):
    """
    Load MultiNLI or SNLI data.
    If the "snli" parameter is set to True, a genre label of snli will be assigned to the data. 
    """
    data = []
    with open(path) as f:
        for line in f:
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                continue
            loaded_example["label"] = LABEL_MAP[loaded_example["gold_label"]]
            if snli:
                loaded_example["genre"] = "snli"
            data.append(loaded_example)
        random.seed(1)
        random.shuffle(data)
    return data

test_snli = load_nli_data("./data/snli_1.0/snli_1.0_dev.jsonl", snli=True)


# In[9]:


def relation_feat(WORD1, WORD2):
    # init
    relation_feat_syn = None
    relation_feat_ant = None
    relation_feat_hype = None
    relation_feat_hypo = None
    relation_feat_same_hype = None
    
    syn_flg = False
    WORD1_syn_lst = wn.synsets(WORD1)
    WORD2_syn_lst =wn.synsets(WORD2)
    if not WORD1_syn_lst or not WORD2_syn_lst:
        return [0,0,0,0,0]
    
    similarity = -np.float('inf')
    sim_flag = False
    for word1_i, word1_syn in enumerate(WORD1_syn_lst):
        for word2_i, word2_syn in enumerate(WORD2_syn_lst):
            # synonymy
            if word1_syn == word2_syn:
                relation_feat_syn = 1
            # antonymy
            if word1_syn.lemmas()[0].antonyms():
                if word1_syn.lemmas()[0].antonyms()[0] == word2_syn.lemmas()[0]:
                    relation_feat_ant = 1
            # hypernymy&hyponymy-prepare
            ## first find their most similarity def
            sim_val = wn.wup_similarity(word1_syn, word2_syn)
            if sim_val: # if they exist similaroty value
                sim_flag = True
                if sim_val > similarity:
                    best_val = sim_val
                    best_syn1 = word1_syn
                    best_syn2 = word2_syn
                    similarity = sim_val
    
    if not relation_feat_syn:
        relation_feat_syn = 0
    if not relation_feat_ant:
        relation_feat_ant = 0     
    if not sim_flag: # in case the 2 words don't exist any similarity definision synset.
        return [0,0,0,0,0]
    
    # hypernymy&hyponymy
    ## base on the def, to figure out their hyper-path
    best_syn1_path = best_syn1.hypernym_paths()
    best_syn2_path = best_syn2.hypernym_paths()

    if relation_feat_syn == 0:
        # to see if the two words have the same hypernym
        for syn1_idx, syn1_path in enumerate(best_syn1_path):
            for syn2_idx, syn2_path in enumerate(best_syn2_path):
                for syn1_hyper_idx, syn1_hyper in enumerate(syn1_path):
                    for syn2_hyper_idx, syn2_hyper in enumerate(syn2_path):
                        # hypernymy&hyponymy
                        if syn1_hyper == best_syn2:
                            edgN = len(syn1_path)-syn1_hyper_idx-1
                            relation_feat_hype = 1-edgN/8
                            
                        if syn2_hyper == best_syn1:
                            edgN = len(syn2_path)-syn2_hyper_idx-1
                            relation_feat_hypo = 1-edgN/8

                        # Same-Hypernym 
                        if syn1_hyper == syn2_hyper:
                            relation_feat_same_hype = 1
        
    if relation_feat_hype:
        relation_feat_hypo = relation_feat_hype
    elif relation_feat_hypo:
        relation_feat_hype = relation_feat_hypo
    else:
        relation_feat_hype = 0
        relation_feat_hypo = 0
        
    if not relation_feat_same_hype:
        relation_feat_same_hype = 0
        
        
    return [relation_feat_syn, relation_feat_ant, relation_feat_hype, relation_feat_hypo, relation_feat_same_hype]

# remove all punctuations (except while space)
def preprocess_word(str_word):
    signtext = string.punctuation 
    signrepl = '@'*len(signtext)
    signtable = str_word.maketrans(signtext,signrepl) 
    return str_word.translate(signtable).replace('@','')

Dic_external_know = {}
for data_i, data_ in enumerate(test_snli):
    print('Process to {}th sentences.'.format(data_i))
    sent_1 = data_['sentence1']
    sent_2 = data_['sentence2'] 
    sent_1 = preprocess_word(sent_1) # remove all punctuations
    sent_2 = preprocess_word(sent_2) # remove all punctuations
    for word1 in sent_1.split():
        for word2 in sent_2.split():
            R_lst = relation_feat(word1, word2) # five relation feature value: [ayn, ant, hype, hypo, same-hype]
            if all(ele == 0 for ele in R_lst):
                l_R = 0 # lr_ij
            else:
                l_R = 1 # lr_ij
            Dic_external_know[(word1,word2)] = l_R # key: (word1, word2); value: L(R_ij) [should be 1 or 0]
    




# In[49]:


import pickle
output = open('snli_dev_relationship_feat.pkl', 'wb')
pickle.dump(Dic_external_know, output)
output.close()


# In[10]:


# Dic_external_know
# get_ipython().system('jupyter nbconvert --to script ProcessWordNet_toy.ipynb')

