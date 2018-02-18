import numpy as np

modelName = "GloVe-1.2/vectors.txt"

def getModel(modelData):

    d = {}

    f = open(modelData)
    for line in f.readlines():
        dim = []
        line = line.strip()
        eles = line.split(' ')
        word = eles[0]
        #print (word)

        for no in eles[1:]:
            dim.append(float(no))

        array = np.asarray(dim)

        #print (array.dtype)

        d[word] = array

    return d

if __name__=='__main__':

    model = getModel(modelName)
    print (model['computer'])