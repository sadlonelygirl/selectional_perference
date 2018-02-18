from embeddingGlove import *

phrases = dataset.dataset('GS2011data.txt')


"""
baseline model(without selectional preference), use element-wise addition for combining subject and object with verb.
"""

def buildPhrasesEmbeddingAdd(phrasesFile):
    eles = buildVerbSubjObj(phrasesFile)
    verb = eles[0]
    d_subjObj_vec = buildEmbeddingSubjObj()
    d_vector = embedding(verb)
    verb_vec = d_vector[verb]

    phrase_vecs = []

    for phrase in phrases:
        eles = phrase.split(' ')
        if eles[0] == verb or eles[3] == verb:
            subj = eles[1]
            obj = eles[2]
            hilo = eles[4]

            subj_vec = d_subjObj_vec[subj]
            obj_vec = d_subjObj_vec[obj]

            phrase_vec = subj_vec+verb_vec+obj_vec

            phraseString = subj+" "+verb+" "+obj+" "+hilo


            #print(phrase_vec.shape)

            phrase_vecs.append((phrase_vec,phraseString))

    return phrase_vecs

"""
baseline model(without selectional preference), use simple element-wise multiplication for combining subject and object with verb.
"""
def buildPhrasesEmbeddingMal(phrasesFile):
    eles = buildVerbSubjObj(phrasesFile)
    verb = eles[0]
    d_subjObj_vec = buildEmbeddingSubjObj()
    d_vector = embedding(verb)
    verb_vec = d_vector[verb]

    phrase_vecs = []

    for phrase in phrases:
        eles = phrase.split(' ')
        if eles[0] == verb or eles[3] == verb:
            subj = eles[1]
            obj = eles[2]
            hilo = eles[4]

            subj_vec = d_subjObj_vec[subj]
            obj_vec = d_subjObj_vec[obj]

            phrase_vec = subj_vec*verb_vec*obj_vec

            phraseString = subj+" "+verb+" "+obj+" "+hilo


            #print(phrase_vec.shape)

            phrase_vecs.append((phrase_vec,phraseString))

    return phrase_vecs

"""
output example:

all phrases of special verb,landmark1, landmark2

provide company service supply 0.808601 HIGH
provide company service leave 0.785844 LOW
provide government cash supply 0.823385 HIGH
provide government cash leave 0.806607 LOW
......

"""

def calculateSimAll(verbPhrases,landMarkPhrases1,landMarkPhrases2, flag):

    #fw=open('resultBaseMal.txt','w')

    if flag=="add":

        verbPhrases1 = buildPhrasesEmbeddingAdd(verbPhrases)
        landMarkPhrases1 = buildPhrasesEmbeddingAdd(landMarkPhrases1)
        landMarkPhrases2 = buildPhrasesEmbeddingAdd(landMarkPhrases2)

    elif flag=="mal":
        verbPhrases1 = buildPhrasesEmbeddingMal(verbPhrases)
        landMarkPhrases1 = buildPhrasesEmbeddingMal(landMarkPhrases1)
        landMarkPhrases2 = buildPhrasesEmbeddingMal(landMarkPhrases2)

    for ele1 in verbPhrases1:
        for ele2 in landMarkPhrases1:
            for ele3 in landMarkPhrases2:
                svo1 = ele1[1].split(' ')
                svo2 = ele2[1].split(' ')
                svo3 = ele3[1].split(' ')

                if svo1[0]==svo2[0]==svo3[0] and svo1[2]==svo2[2]==svo3[2]:
                    #print (ele1[1])
                    #print (ele2[1])
                    #print (ele3[1])
                    #print (calculateSim(ele1[0],ele2[0]))
                    #print (calculateSim(ele1[0],ele3[0]))
                    #print ()

                    verb = ele1[1].split(' ')[1]
                    subj = ele1[1].split(' ')[0]
                    obj = ele1[1].split(' ')[2]

                    landMark1 = ele2[1].split(' ')[1]
                    landMark2 = ele3[1].split(' ')[1]

                    score1 = calculateSim(ele1[0],ele2[0],'dot')
                    score2 = calculateSim(ele1[0],ele3[0],'dot')

                    hilo1 = ele2[1].split(' ')[3]
                    hilo2 = ele3[1].split(' ')[3]

                    #fw.write(verb+' '+subj+' '+obj+' '+landMark1+' '+str(score1)+' '+hilo1+'\n')
                    #fw.write(verb+' '+subj+' '+obj+' '+landMark2+' '+str(score2)+' '+hilo2+'\n')

                    print (verb+' '+subj+' '+obj+' '+landMark1+' '+str(score1)+' '+hilo1)
                    print (verb+' '+subj+' '+obj+' '+landMark2+' '+str(score2)+' '+hilo2)





if __name__=='__main__':
    calculateSimAll("phrases/"+sys.argv[1],"phrases/"+sys.argv[2],"phrases/"+sys.argv[3],sys.argv[4])