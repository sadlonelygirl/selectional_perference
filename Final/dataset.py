def dataset(txtFile):

    f = open(txtFile,'r')
    lines = f.readlines()

    phrases = set([])

    for line in lines:
        line = line.strip()
        eles = line.split()
        phrase = eles[1]+" "+eles[2]+" "+eles[3]+" "+eles[4]+" "+eles[6]
        phrases.add(phrase)

    phrases = list(phrases)
    phrases.sort()

    return phrases

if __name__=='__main__':

    dataset('GS2011data.txt')