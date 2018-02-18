import numpy as np

def result(resultFile):
	file = open(resultFile)
	allLinesDup = file.readlines()
	allLines = []

	for i in range(len(allLinesDup)):
		if i%4==2 or i%4==3:
			allLines.append(allLinesDup[i])

	#print allLines

	c_high = 0
	c_low = 0
	high = 0.
	low = 0.
	for line in allLines:
		elements = line.split()
		if elements[5] == 'HIGH':
			high += float(elements[4])
			c_high += 1
		else:
			low += float(elements[4])
			c_low += 1
	return high/c_high, low/c_low,

def oneHumanAverage(verb, subj, obj, landmark, hilo):
	file = open('GS2011data.txt')
	allLines = file.readlines()
	sum = 0.
	count = 0
	for line in allLines:
		elements = line.split()
		if elements[1]==verb and elements[2]==subj and elements[3]==obj and elements[4]==landmark and elements[6]==hilo:
			sum += float(elements[5])
			count += 1
	return sum/count

def modelAndHumanAverage(resFile):

	file = open(resFile)
	allLinesDup = file.readlines()
	allLines = []

	for i in range(len(allLinesDup)):
		if i%4==2 or i%4==3:
			allLines.append(allLinesDup[i])

	model = []
	human = []
	for line in allLines:
		elements = line.split()
		model.append(elements[4])
		humanAverage = oneHumanAverage(elements[0],elements[1],elements[2],elements[3],elements[5])
		human.append(humanAverage)
	model2 = sorted(model)
	human2 = sorted(human)
	modelRank = []
	humanRank = []
	for ele in model:
		index = model2.index(ele)
		modelRank.append(index)
	for ele in human:
		index = human2.index(ele)
		humanRank.append(index)
	return modelRank, humanRank

def rho(values1, values2):
	""" Computes Spearman's rho for two lists of observations with no ties

		Parameters:
			values1:	list of observations
			values2:	list of observations
	"""
	v1, v2 = np.array(values1), np.array(values2)
	n = len(v1)
	return 1 - 6 * np.sum((v1 - v2)**2) / float(n * (n*n -1))

if __name__=='__main__':

	print "selectional preference, dot product, word2vec: "

	print "high, low: " + str(result('ResDotW2V/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResDotW2V/resAll.txt')
	print "rho: " + str(rho(modelHuman[0], modelHuman[1]))

	print

	print "selectional preference, multiplication, word2vec: "

	print result('ResMalW2V/resAll.txt')
	modelHuman = modelAndHumanAverage('ResMalW2V/resAll.txt')
	print rho(modelHuman[0], modelHuman[1])

	print
	
	print "selectional preference, dot product, GloVe: "

	print "high, low: " + str(result('ResDotGlove/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResDotGlove/resAll.txt')
	print "rho: " + str(rho(modelHuman[0], modelHuman[1]))

	print
	print "selectional preference, multiplication, GloVe:"

	print "high, low: " + str(result('ResMalGlove/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResMalGlove/resAll.txt')
	print "rho: "+ str(rho(modelHuman[0], modelHuman[1]))

	print
	print "baseline, addition, word2vec: "

	print "high, low: " + str(result('ResBaseAddW2V/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResBaseAddW2V/resAll.txt')
	print "rho: " + str(rho(modelHuman[0], modelHuman[1]))

	print
	print "baseline, multiplication, word2vec: "

	print "high, low: " + str(result('ResBaseMalW2V/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResBaseMalW2V/resAll.txt')
	print "rho:" + str(rho(modelHuman[0], modelHuman[1]))

	print
	print "baseline, addition, GloVe: "

	print "high, low: " + str(result('ResBaseAddGlove/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResBaseAddGlove/resAll.txt')
	print "rho:" +str(rho(modelHuman[0], modelHuman[1]))

	print
	print "baseline, multiplication, GloVe: "

	print "high, low: " + str(result('ResBaseMalGlove/resAll.txt'))
	modelHuman = modelAndHumanAverage('ResBaseMalGlove/resAll.txt')
	print "rho:" + str(rho(modelHuman[0], modelHuman[1]))


