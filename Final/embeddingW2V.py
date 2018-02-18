from nltk.corpus import wordnet as wn
import gensim.models as md
import numpy as np
import dataset
import collections
import copy
from scipy import spatial
import sys

groups = ["phenomenon","object","possession","state","communication","body","relation","feeling","Tops","attribute","quantity","plant","cognition","location","event","food","motive","group","substance","person","animal","act","artifact","time","shape","process"]
phrases = dataset.dataset('GS2011data.txt')
#print (len(groups))
modelName = '../../sentiment/CNN/GoogleNews-vectors-negative300.bin'
model = md.Word2Vec.load_word2vec_format(modelName,binary=True)


"""
get verb and selectional preference, use all transitive phrases from NBC, return a verb string, subject and obj dictionary with weight of different groups

for example, obj word(once), has 9 synsets, 7 belong to communication, 1 belong to quality, 1 belong to person, then 7/9 communication, 1/9 quality, 1/9 person.

"""

def buildVerbSubjObj(phrasesFile):

    d_subj = collections.OrderedDict()
    for ele in groups:
        d_subj[ele] = 0
    d_obj = collections.OrderedDict()
    for ele in groups:
        d_obj[ele] = 0

    f = open(phrasesFile,'r')
    lines = f.readlines()

    verb = lines[0].split(' ')[1]

    for line in lines:

        eles = line.split(' ')

        #subjGroup = set([])
        #objGroup = set([])

        subjGroup = []
        objGroup = []

        subj = eles[0]
        obj = eles[2].strip()

        for synset in wn.synsets(subj):
            group = synset.lexname()
            if group.startswith('noun.'):
                subjGroup.append(group.split('.')[1])

        for ele in subjGroup:
            d_subj[ele] += 1/float(len(subjGroup))

        for synset in wn.synsets(obj):
            group = synset.lexname()
            if group.startswith('noun.'):
                objGroup.append(group.split('.')[1])

        for ele in objGroup:
            d_obj[ele] += 1/float(len(objGroup))

    d_subj_sorted = sorted(d_subj.items(), key=lambda d:d[1], reverse = True)
    d_obj_sorted = sorted(d_obj.items(), key=lambda d:d[1], reverse = True)

    #print (d_subj_sorted)
    #print (d_obj_sorted)

    return verb,d_subj,d_obj


def buildEmbeddingSubjObj():

    d_subjObj_vec = {}
    subjobj = set([])

    for phrase in phrases:
        eles = phrase.split(' ')
        subj = eles[1]
        obj = eles[2]

        subjobj.add(subj)
        subjobj.add(obj)

    for word in subjobj:
        try:
            vector = model[word]
            d_subjObj_vec[word] = vector
        except KeyError:
            vector = np.random.uniform(-0.23, 0.23, [1,300])
            d_subjObj_vec[word] = vector

    return d_subjObj_vec


def embedding(verb):

    VerbAndGroups = copy.copy(groups)
    VerbAndGroups.append(verb)

    d_vector = {}

    for word in VerbAndGroups:
        try:
            vector = model[word]
            d_vector[word] = vector
        except KeyError:
            vector = np.random.uniform(-0.23, 0.23, [1,300])
            d_vector[word] = vector

    #print (d_vector)
    return d_vector

"""
build verb embedding, extend 300*1 to 300*53
"""

def buildVerbEmbedding(phrasesFile):

    eles = buildVerbSubjObj(phrasesFile)

    verb = eles[0]
    d_subj = eles[1]
    d_obj = eles[2]

    d_vector = embedding(verb)

    verb_vec = d_vector[verb]
    #print (verb_vec.shape)

    for subj in d_subj:
        verb_vec = np.vstack((verb_vec,d_vector[subj]*d_subj[subj]))

    #print (verb_vec.shape)

    for obj in d_obj:
        verb_vec = np.vstack((verb_vec,d_vector[obj]*d_obj[obj]))

    #print (verb_vec)
    #print (verb_vec.shape)
    return verb_vec

"""
extend 300*1 to (300*53)*1
"""
"""
def buildVerbEmbeddingV2(phrasesFile):


    eles = buildVerbSubjObj(phrasesFile)

    verb = eles[0]
    d_subj = eles[1]
    d_obj = eles[2]

    d_vector = embedding(verb)

    verb_vec = d_vector[verb]

    for subj in d_subj:
        #print ("subj: "+ subj)
        verb_vec = np.append(verb_vec,d_vector[subj]*d_subj[subj])

    for obj in d_obj:
        #print ("obj: "+ obj)
        verb_vec = np.append(verb_vec,d_vector[obj]*d_obj[obj])

    return verb_vec

"""

"""
build phrase embedding, use dot product as similarity measure for combining subj/obj with verb.
"""

def buildPhrasesEmbedding(phrasesFile):
    eles = buildVerbSubjObj(phrasesFile)
    verb = eles[0]
    d_subjObj_vec = buildEmbeddingSubjObj()
    verb_vec = buildVerbEmbedding(phrasesFile)

    phrase_vecs = []

    for phrase in phrases:
        eles = phrase.split(' ')
        if eles[0] == verb or eles[3] == verb:
            subj = eles[1]
            obj = eles[2]
            hilo = eles[4]

            subj_vec = d_subjObj_vec[subj]
            obj_vec = d_subjObj_vec[obj]

            phrase_vec = verb_vec[1]

            phraseString = subj+" "+verb+" "+obj+" "+hilo

            for groupSubj in verb_vec[1:27]:
                if groupSubj.shape != subj_vec.shape:
                    subj_vec = subj_vec.reshape(groupSubj.shape)

                dotValue = np.dot(groupSubj,subj_vec)
                groupDim =  np.asarray([dotValue])

                phrase_vec = np.append(phrase_vec,groupDim)

            for groupObj in verb_vec[27:]:
                if groupObj.shape != obj_vec.shape:
                    obj_vec = obj_vec.reshape(groupObj.shape)
                dotValue = np.dot(groupObj,obj_vec)
                groupDim =  np.asarray([dotValue])
                phrase_vec = np.append(phrase_vec,groupDim)

            phrase_vecs.append((phrase_vec,phraseString))

    return phrase_vecs

"""
build phrase embedding, use element-wise multiplication as similarity measure for combining subj/obj with verb.
"""
def buildPhrasesEmbeddingV2(phrasesFile):

    eles = buildVerbSubjObj(phrasesFile)
    verb = eles[0]
    d_subjObj_vec = buildEmbeddingSubjObj()
    verb_vec = buildVerbEmbedding(phrasesFile)

    phrase_vecs = []

    for phrase in phrases:
        eles = phrase.split(' ')
        if eles[0] == verb or eles[3] == verb:
            subj = eles[1]
            obj = eles[2]
            hilo = eles[4]

            subj_vec = d_subjObj_vec[subj]
            obj_vec = d_subjObj_vec[obj]

            phrase_vec = verb_vec[1]

            phraseString = subj+" "+verb+" "+obj+" "+hilo

            for groupSubj in verb_vec[1:27]:
                if groupSubj.shape != subj_vec.shape:
                    subj_vec = subj_vec.reshape(groupSubj.shape)

                mal_vec = groupSubj*subj_vec
                phrase_vec = np.vstack((phrase_vec,mal_vec))

            for groupObj in verb_vec[27:]:
                if groupObj.shape != obj_vec.shape:
                    obj_vec = obj_vec.reshape(groupObj.shape)
                mal_vec = groupObj*obj_vec
                phrase_vec = np.vstack((phrase_vec,mal_vec))

            phrase_vecs.append((phrase_vec,phraseString))

    return phrase_vecs

"""
calculate cosine similarity.
"""

def calculateSim(phrase_vec1, phrase_vec2, flag):

    if flag=='multiplication':
        phrase_vec1=phrase_vec1.reshape(phrase_vec1.shape[0]*phrase_vec1.shape[1])
        phrase_vec2=phrase_vec2.reshape(phrase_vec2.shape[0]*phrase_vec2.shape[1])

    cosine_similarity = 9999

    try:
        cosine_similarity = np.dot(phrase_vec1, phrase_vec2)/(np.linalg.norm(phrase_vec1)* np.linalg.norm(phrase_vec2))
    except ValueError:
        #print (phrase_vec1.shape)
        #print (phrase_vec2.shape)
        pass

    return cosine_similarity

"""
output example:

all phrases of special verb,landmark1, landmark2

provide company service supply 0.808601 HIGH
provide company service leave 0.785844 LOW
provide government cash supply 0.823385 HIGH
provide government cash leave 0.806607 LOW
......

"""

def calculateSimAll(verbPhrases,landMarkPhrases1,landMarkPhrases2,flag):

    #fw = open('resultDot.txt','w')

    if flag=="dot":
        verbPhrases1 = buildPhrasesEmbedding(verbPhrases)
        landMarkPhrases1 = buildPhrasesEmbedding(landMarkPhrases1)
        landMarkPhrases2 = buildPhrasesEmbedding(landMarkPhrases2)

    elif flag=='multiplication':
        verbPhrases1 = buildPhrasesEmbeddingV2(verbPhrases)
        landMarkPhrases1 = buildPhrasesEmbeddingV2(landMarkPhrases1)
        landMarkPhrases2 = buildPhrasesEmbeddingV2(landMarkPhrases2)

    for ele1 in verbPhrases1:
        for ele2 in landMarkPhrases1:
            for ele3 in landMarkPhrases2:
                svo1 = ele1[1].split(' ')
                svo2 = ele2[1].split(' ')
                svo3 = ele3[1].split(' ')

                if svo1[0]==svo2[0]==svo3[0] and svo1[2]==svo2[2]==svo3[2]:

                    verb = ele1[1].split(' ')[1]
                    subj = ele1[1].split(' ')[0]
                    obj = ele1[1].split(' ')[2]

                    landMark1 = ele2[1].split(' ')[1]
                    landMark2 = ele3[1].split(' ')[1]

                    score1 = calculateSim(ele1[0],ele2[0],flag)
                    score2 = calculateSim(ele1[0],ele3[0],flag)

                    hilo1 = ele2[1].split(' ')[3]
                    hilo2 = ele3[1].split(' ')[3]

                    #fw.write(verb+' '+subj+' '+obj+' '+landMark1+' '+str(score1)+' '+hilo1+'\n')
                    #fw.write(verb+' '+subj+' '+obj+' '+landMark2+' '+str(score2)+' '+hilo2+'\n')

                    print (verb+' '+subj+' '+obj+' '+landMark1+' '+str(score1)+' '+hilo1)
                    print (verb+' '+subj+' '+obj+' '+landMark2+' '+str(score2)+' '+hilo2)

"""
def buildSubjObjEmbedding(phrasesFile):

    eles = buildVerbSubjObj(phrasesFile)
    verb = eles[0]
    d_vector = embedding(verb)

    subjObjPairs = []

    for phrase in phrases:
        eles = phrase.split(' ')
        if eles[0] == verb or eles[3] == verb:
            #print (phrase)
            subj = eles[1]
            obj = eles[2]
            #print (subj)

            subjGroup = set([])
            objGroup = set([])

            for synset in wn.synsets(subj):
                group = synset.lexname()
                if group.startswith('noun.'):
                    subjGroup.add(group.split('.')[1])

            #print (subjGroup)
            #print (obj)

            for synset in wn.synsets(obj):
                group = synset.lexname()
                if group.startswith('noun.'):
                    objGroup.add(group.split('.')[1])

            #print (objGroup)
            #print ()

            d_subj = collections.OrderedDict()
            for ele in groups:
                d_subj[ele] = 0
            d_obj = collections.OrderedDict()
            for ele in groups:
                d_obj[ele] = 0

            for ele in subjGroup:
                d_subj[ele] += 1

            for ele in objGroup:
                d_obj[ele] += 1

            #print (d_subj)
            #print (d_obj)

            subj_vect = np.zeros(100)

            #print (len(d_subj))
            #print (len(groups))

            for subj_e in d_subj:
                if d_subj[subj_e] == 0:
                    subj_vect = np.append(subj_vect,np.zeros(100))
                else:
                    subj_vect = np.append(subj_vect,d_vector[subj_e])

            for obj_e in d_obj:
                subj_vect = np.append(subj_vect,np.zeros(100))

            #print (subj_vect)
            #print (subj_vect.shape)

            obj_vect = np.zeros(100)

            for subj_e in d_subj:
                obj_vect = np.append(obj_vect,np.zeros(100))

            for obj_e in d_obj:
                if d_obj[obj_e] == 0:
                    obj_vect = np.append(obj_vect,np.zeros(100))
                else:
                    obj_vect = np.append(obj_vect,d_vector[obj_e])

            #print (obj_vect)
            #print (obj_vect.shape)

            #print (subj)

            subjObjPairs.append((subj,obj,subj_vect,obj_vect, subjGroup, objGroup))

    return subjObjPairs

def buildPhrasesEmbedding(phrasesFile):

    subjObjPairs = buildSubjObjEmbedding(phrasesFile)
    verb_vec = buildVerbEmbedding(phrasesFile)

    for ele in subjObjPairs:
        subj = ele[0]
        obj = ele[1]
        subj_vect = ele[2]
        obj_vect = ele[3]
        phrase_vec = subj_vect+verb_vec+obj_vect

        print (subj)
        print (obj)
        print (phrase_vec)
        print (phrase_vec.shape)

"""
if __name__=='__main__':

    #np.set_printoptions(threshold=10000)

    #verb_vec = buildVerbEmbedding("spellPhrases")

    #print (verb_vec)
    #print (verb_vec.shape)
    #eles = buildSubjObjEmbedding("spellPhrases")

    #print (eles[0])

    #for ele in eles:
        #print (ele)

    #buildPhrasesEmbedding("spellPhrases")

    #calculateSimAll("VerbPhrases/src/phrases/writePhrases","VerbPhrases/src/phrases/spellPhrases","VerbPhrases/src/phrases/publishPhrases")

    calculateSimAll("phrases/"+sys.argv[1],"phrases/"+sys.argv[2],"phrases/"+sys.argv[3], sys.argv[4])

    #buildPhrasesEmbeddingV2("spellPhrases")

    #buildVerbSubjObj("publishPhrases")

    #embedding("spell")

    #buildVerbEmbedding("spellPhrases")
    #buildPhrasesEmbedding("spellPhrases")

    #buildEmbeddingVocabulary()

    #buildEmbeddingSubjObj()

