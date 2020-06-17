# -*- coding: UTF-8 -*-
import numpy as np
import re
import random 
 
"""
createVocabList函数功能: 将切分的实验样本词条整理成不重复的词条列表，也就是词汇表
形参:
    dataSet - 整理的样本数据集(全部)，类似一维数组，每个元素是一个文件的所有字符
返回值:
    vocabSet - 返回不重复的词条列表，也就是词汇表
"""
def createVocabList(dataSet):
    vocabSet = set([])  # 创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # 取并集
    return list(vocabSet)
  
 
"""
bagOfWords2VecMN函数功能: 根据vocabList词汇表，构建词带模型
形参:
    vocabList - createVocabList返回的列表
    inputSet - 切分的词条列表，一个文件的所有符号
返回值:
    returnVec - 文档向量,词袋模型
"""
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)  # 创建一个其中所含元素都为0的向量
    for word in inputSet:             # 遍历每个词条
        if word in vocabList:         # 如果词条存在于词汇表中，则计数加一。如果词汇表中有些词出现次数为0，也是可以的
            returnVec[vocabList.index(word)] += 1 #index可以查找word在vocabList里面的位置
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec  # 返回词袋模型
 
 
"""
trainNB0函数功能: 朴素贝叶斯分类器训练函数
形参:
    trainMatrix - 训练集文档矩阵，即bagOfWords2Vec返回的returnVec构成的矩阵，全是数字（字符出现次数）而不是字符了
    trainCategory - 训练集类别标签向量，即loadDataSet返回的classVec
返回值:
    p0Vect - 正常邮件类的条件概率数组
    p1Vect - 垃圾邮件类的条件概率数组
    pAbusive - 文档属于垃圾邮件类的概率
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)  		# 计算用于训练的文档数目
    numWords = len(trainMatrix[0])  		# 计算每篇文档的词条数，假设每个邮件都是一样的单词总数，不足就补0
    pAbusive = sum(trainCategory) / float(numTrainDocs)  # 文档属于垃圾邮件类的概率，为了这一步，才把邮件分类为0/1
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)  			
    # 创建numpy.ones数组,假设每个词都出现了1次,防止因为单词出现次数为0导致概率为0，特征丢失
    p0Denom = 1.0+numTrainDocs
    p1Denom = 1.0+numTrainDocs  				# 拉普拉斯平滑
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:  		
            p1Num += trainMatrix[i] 		# 矩阵相加，求垃圾邮件中每个单词出现的总次数
            p1Denom += sum(trainMatrix[i])	# 一个数，垃圾邮件的总次数
        else:  					
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 取对数，更利于计算，也防止下溢
    return (np.log(p0Num / p0Denom)), (np.log(p1Num / p1Denom)), pAbusive  
    # 返回属于正常邮件类的条件概率（取对数）数组，属于垃圾邮件类的条件概率（取对数）数组，文档属于垃圾邮件类的概率
 
 
"""
classifyNB函数功能: 朴素贝叶斯分类器分类函数
形参:
	vec2Classify - 待分类的文本词带模型
	p0Vec - 正常邮件类的条件概率数组
	p1Vec - 垃圾邮件类的条件概率数组
	pClass1 - 文档属于垃圾邮件的概率
返回值:
	0 - 正常邮件
	1 - 垃圾邮件
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #‘*’是对应位置相乘，相当于求内积，取对数后只需要用加法，前面是先利用向量，减少了循环
    if (sum(vec2Classify*p1Vec)+np.log(pClass1)) > (sum(vec2Classify*p0Vec)+np.log(1.0-pClass1)):
        return 1
    else:
        return 0

"""
textParse函数功能: 将字符串解析为字符串列表
形参：
	bigString - 从文件中读取的数据
"""
def textParse(bigString):  # 将字符串转换为字符列表
    listOfTokens = re.split(r'\W+', bigString)  
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字,匹配非字母、非数字一次或多次
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]  # 除了单个字母，例如大写的I，其它单词变成小写


docList = []
classList = []
fullText = []
for i in range(1, 31):  # 读取60个txt文件（垃圾文件30个，非垃圾文件30个），生成符号数组
    wordList = textParse(open('D:/bayes/spam/%d.txt' % i, 'r').read())  # 读取每个垃圾邮件，并字符串转换成字符串列表
    docList.append(wordList)
    fullText.append(wordList)
    classList.append(1)  # 标记垃圾邮件，1表示垃圾文件
    wordList = textParse(open('D:/bayes/ham/%d.txt' % i, 'r').read())  # 读取每个非垃圾邮件，并字符串转换成字符串列表
    docList.append(wordList)
    fullText.append(wordList)
    classList.append(0)  # 标记正常邮件，0表示正常文件
errorrate=[]
vocabList = createVocabList(docList)  # 创建不重复的词汇表
#print(vocabList)   #查看这个数据集的词汇表
#train=bagOfWords2VecMN(vocabList,docList[3])
#print(train)   #查看词袋模型的具体表现
for x in range(100):
    trainingSet = list(range(60))
    testSet = []  # 创建存储训练集的索引值的列表和测试集的索引值的列表
    for i in range(20):  # 从60个邮件中，随机挑选出40个作为训练集,20个做测试集
        randIndex = int(random.uniform(0, len(trainingSet)))  # 随机选取索引值
        testSet.append(trainingSet[randIndex])  # 添加测试集的索引值
        del (trainingSet[randIndex])  # 在训练集列表中删除添加到测试集的索引值
    trainMat = []	# 训练集的词带矩阵，每一个元素不再是符号而是该词出现的次数，每一行代表一个训练文件的特征向量（词带）
    trainClasses = []  	# 创建训练集类别标签系向量
    for docIndex in trainingSet:  # 遍历训练集
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))  # 将生成的词带模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])  # 将类别添加到训练集类别标签系向量中
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))  # 根据训练集求概率
    errorCount = 0  # 错误分类计数
    for docIndex in testSet:  # 遍历测试集
        if classifyNB(np.array(bagOfWords2VecMN(vocabList, docList[docIndex])), p0V, p1V, pSpam) != classList[docIndex]:  # 如果分类错误
            errorCount += 1  # 错误计数加1
    errorrate.append((float(errorCount) / len(testSet) * 100))
print('错误率：%.2f%%' % ((sum(errorrate)/100)))
