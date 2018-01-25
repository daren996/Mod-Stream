# -*- coding: utf-8 -*-
import codecs
import json
from sklearn import metrics
import pylab as py
import numpy as np
import math

def MeanAndVar(dataList):
    mean = sum(dataList)*1.0 / len(dataList)
    varience = math.sqrt(sum((mean - value) ** 2 for value in dataList)*1.0 / len(dataList))
    return (mean, varience)

class ClusterEvaluation():
    def __init__(self, resultFilePath):
        self.tweetsCleaned = []
        self.resultFilePath = resultFilePath
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []
        self.SCList = []
        
        self.AMITopicKList = []
        self.NMITopicKList = []
        self.MITopicKList = []
        self.ARITopicKList = []
        self.homogeneityTopicKList = []
        self.completenessTopicKList = []
        self.VTopicKList = []
        self.SCTopicKList = []
        self.docNum = 0

        self.labelsPred = {}

    # Get evaluation of each sample
    def evaluatePerSample(self, sampleNo):
        labelsTrue = []
        labelsPred = []
        for d in self.docs:
            documentID = d[0]
            if documentID in self.labelsPred:
                labelsTrue.append(self.labelsTrue[documentID])
                labelsPred.append(self.labelsPred[documentID])
        AMI = metrics.adjusted_mutual_info_score(labelsTrue, labelsPred)
        NMI = metrics.normalized_mutual_info_score(labelsTrue, labelsPred)
        MI = metrics.mutual_info_score(labelsTrue, labelsPred)
        ARI = metrics.adjusted_rand_score(labelsTrue, labelsPred)
        homogeneity = metrics.homogeneity_score(labelsTrue, labelsPred)
        completeness = metrics.completeness_score(labelsTrue, labelsPred)
        V = metrics.v_measure_score(labelsTrue, labelsPred)
#        SC = metrics.silhouette_score(self.X, self.labelsPred, metric='sqeuclidean') #Silhouette Coefficient
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)
        #        self.SCList.append(SC)

    # Get mean and var of all evaluation
    def evaluateAllSamples(self, K):
        self.ARITopicKList.append(MeanAndVar(self.ARIList))
        self.MITopicKList.append(MeanAndVar(self.MIList))
        self.AMITopicKList.append(MeanAndVar(self.AMIList))
        self.NMITopicKList.append(MeanAndVar(self.NMIList))
        self.homogeneityTopicKList.append(MeanAndVar(self.homogeneityList))
        self.completenessTopicKList.append(MeanAndVar(self.completenessList))
        self.VTopicKList.append(MeanAndVar(self.VList))
        
        self.AMIList = []
        self.NMIList = []
        self.MIList = []
        self.ARIList = []
        self.homogeneityList = []
        self.completenessList = []
        self.VList = []

    def drawEvaluationResult(self, KRange, Xlabel, titleStr):
        ARIVarianceList = [item[1] for item in self.ARITopicKList]
#        MIVarianceList = [item[1] for item in self.MITopicKList]
        AMIVarianceList = [item[1] for item in self.AMITopicKList]
        NMIVarianceList = [item[1] for item in self.NMITopicKList]
        homogeneityVarianceList = [item[1] for item in self.homogeneityTopicKList]
        completenessVarianceList = [item[1] for item in self.completenessTopicKList]
        VVarianceList = [item[1] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRange/iterNumRange:' + repr(KRange) + '\n')
            fout.write('ARIVarianceList:' + repr(ARIVarianceList) + '\n')
            fout.write('AMIVarianceList:' + repr(AMIVarianceList) + '\n')
            fout.write('NMIVarianceList:' + repr(NMIVarianceList) + '\n')
            fout.write('homogeneityVarianceList:' + repr(homogeneityVarianceList) + '\n')
            fout.write('completenessVarianceList:' + repr(completenessVarianceList) + '\n')
            fout.write('VVarianceList:' + repr(VVarianceList) + '\n')
            fout.write('\n')

        ARIMeanList = [item[0] for item in self.ARITopicKList]
#        MIMeanList = [item[0] for item in self.MITopicKList]
        AMIMeanList = [item[0] for item in self.AMITopicKList]
        NMIMeanList = [item[0] for item in self.NMITopicKList]
        homogeneityMeanList = [item[0] for item in self.homogeneityTopicKList]
        completenessMeanList = [item[0] for item in self.completenessTopicKList]
        VMeanList = [item[0] for item in self.VTopicKList]
        with open(self.resultFilePath, 'a') as fout:
            fout.write('KRange/iterNumRange:' + repr(KRange) + '\n')
            fout.write('ARIMeanList:' + repr(ARIMeanList) + '\n')
            fout.write('AMIMeanList:' + repr(AMIMeanList) + '\n')
            fout.write('NMIMeanList:' + repr(NMIMeanList) + '\n')
            fout.write('homogeneityMeanList:' + repr(homogeneityMeanList) + '\n')
            fout.write('completenessMeanList:' + repr(completenessMeanList) + '\n')
            fout.write('VMeanList:' + repr(VMeanList) + '\n')
            fout.write('\n')
            
        with open(self.resultFilePath, 'a') as fout:
            fout.write('\n#%s\n' % (titleStr)) 
            fout.write('\n\n#ARI\n')  
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(ARIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], ARIMeanList[j], ARIVarianceList[j]))
            fout.write('\n\n#AMI\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(AMIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], AMIMeanList[j], AMIVarianceList[j]))
            fout.write('\n\n#NMI\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(NMIMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], NMIMeanList[j], NMIVarianceList[j]))
            fout.write('\n\n#homogeneity\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(homogeneityMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], homogeneityMeanList[j], homogeneityVarianceList[j]))
            fout.write('\n\n#completeness\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(completenessMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], completenessMeanList[j], completenessVarianceList[j]))
            fout.write('\n\n#V\n') 
            fout.write('#X\tMean\tSD\n') 
            for j in range(len(VMeanList)):
                fout.write('%f\t%.4f\t%.4f\n' % (KRange[j], VMeanList[j], VVarianceList[j]))

        p1 = py.plot(KRange, ARIMeanList, 'r-*')
        p2 = py.plot(KRange, AMIMeanList, 'g-D')
        p3 = py.plot(KRange, NMIMeanList, 'g-s')
        p4 = py.plot(KRange, homogeneityMeanList, 'b-v')
        p5 = py.plot(KRange, completenessMeanList, 'b-^')
        p6 = py.plot(KRange, VMeanList, 'r-s')
        py.errorbar(KRange, ARIMeanList, yerr=ARIVarianceList, fmt='r*')
        py.errorbar(KRange, AMIMeanList, yerr=AMIVarianceList, fmt='gD')
        py.errorbar(KRange, NMIMeanList, yerr=NMIVarianceList, fmt='gs')
        py.errorbar(KRange, homogeneityMeanList, yerr=homogeneityVarianceList, fmt='bv')
        py.errorbar(KRange, completenessMeanList, yerr=completenessVarianceList, fmt='b^')
        py.errorbar(KRange, VMeanList, yerr=VVarianceList, fmt='rs')
        
        py.legend([p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]], ['ARI', 'AMI', 'NMI', 'Homogeneity','Completeness', 'V'])
        py.xlabel(Xlabel)
        py.ylabel('Performance')      
        py.title(titleStr)
        py.grid(True)
        py.show()

    # Get labelsPred.
    def getMStreamPredLabels(self, inFile):
        with codecs.open(inFile, 'r') as fin:
            for line in fin:
                try:
                    documentID = line.strip().split()[0]
                    clusterNo = line.strip().split()[1]
                    self.labelsPred[int(documentID)] = int(clusterNo)
                except:
                    print(line)

    # Get labelsTrue and docs.
    def getMStreamTrueLabels(self, inFile, dataset):
        self.labelsTrue = {}
        self.docs = []
        outFile = inFile + "Full.txt"
        with codecs.open(dataset, 'r') as fin:
            for docJson in fin:
                try:
                    docObj = json.loads(docJson)
                    self.labelsTrue[int(docObj['tweetId'])] = int(docObj['clusterNo'])
                    self.docs.append([int(docObj['tweetId']), docObj['textCleaned']])
                    # clusterNames.append(docObj['clusterName'])
                except:
                    print(docJson)

        with codecs.open(outFile, 'a') as fout:
            for i in range(len(self.docs)):
                docObj = {}
                documentID = self.docs[i][0]
                if documentID in self.labelsPred:
                    docObj['trueCluster'] = self.labelsTrue[documentID]
                    docObj['predictedCluster'] = self.labelsPred[documentID]
                    docObj['text'] = self.docs[i][1]
#                    docObj['clusterName'] = clusterNames[i]
                    docJson = json.dumps(docObj)
                    fout.write(docJson + '\n')

    def getMStreamKPredNum(self, inFile):
        labelsPred = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
        KPredNum = np.unique(labelsPred).shape[0]
        return KPredNum

    # Get scaled cluster number and scaled document number. Kthreshold is minimum number of a scaled cluster.
    def getPredNumThreshold(self, inFile, Kthreshold):
        KPredNum = 0
        docRemainNum = 0
        docTotalNum = 0
        with codecs.open(inFile, 'r', 'utf-8') as fin:
            clusterSizeStr = fin.readline().strip().strip(',')
        clusterSizeList = clusterSizeStr.split(',\t')
        for clusterSize in clusterSizeList:
            try:
                clusterSizeCouple = clusterSize.split(':')
                docTotalNum += int(clusterSizeCouple[1])
                if int(clusterSizeCouple[1]) > Kthreshold:
                    KPredNum += 1
                    docRemainNum += int(clusterSizeCouple[1])
            except:
                pass
        return (KPredNum,docRemainNum,docTotalNum)

    def docNumPerCluster(self, inFile, parameter, sampleNo, resultFile):
        labelsPred = []
        labelsTrue = []
        with codecs.open(inFile, 'r') as fin:
            for lineJson in fin:
                resultObj = json.loads(lineJson)
                labelsPred.append(resultObj['predictedCluster'])
                labelsTrue.append(resultObj['trueCluster'])
        topicNoVec, indices = np.unique(labelsPred, return_inverse=True)
        # topicNoVec stores the topicNos, sorted by the number of documents in these topics.
        docToTopicVec = topicNoVec[indices]
        # docToTopicVec stores the topicNo for each document.
        
        predTopicNum = len(topicNoVec) 
        maxTopicNo = max(topicNoVec)
        clusterNum = np.unique(self.labelsTrue).shape[0] #Obtain the true number of clusters.
        docNum = len(labelsPred)
        topicSizeVec = [0 for i in range(predTopicNum)]
        for topicNo in indices:
            topicSizeVec[topicNo] += 1

        topicsVec = []
        for topicNo in range(predTopicNum):
            topicTrueNo = topicNoVec[topicNo] #the topicNo in the result file.
            topicsVec.append((topicTrueNo, topicSizeVec[topicNo]))
        topicsVec.sort(key=lambda tup: tup[1], reverse=True)

            
        #find the first and second cluster that each topic relates to.
        topicClusterVec = []
        for i in range(predTopicNum):
            topicNo = topicsVec[i][0] #From topics with most clusters.
            clusterVec = [[clusterNo, 0] for clusterNo in range(clusterNum+1)] 
            #stores the clusters and the number of its documents for each topic.
            for docNo in range(docNum):
                if labelsPred[docNo] == topicNo:
                    clusterNo = int(labelsTrue[docNo])
                    clusterVec[clusterNo][1] += 1
            clusterVec.sort(key=lambda lis: lis[1], reverse=True)
            topicClusterVec.append(clusterVec)
        
        with codecs.open(resultFile, 'a') as fout:
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('topicNo:docNum ')
            for tup in topicsVec:
                fout.write(str(tup[0]) + ':' + str(tup[1]) + ', ')
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('FirClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[0][0], clusterVec[0][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('SecClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[1][0], clusterVec[1][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('predTopicNum=' + str(predTopicNum) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('ThiClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[2][0], clusterVec[2][1]))        
            fout.write('\n')
            
            fout.write('parameter=' + str(parameter) + ' ')
            fout.write('sampleNo=' + str(sampleNo) + ' ')
            fout.write('FouClus:docNum ')     
            for clusterVec in topicClusterVec:
                fout.write('%d:%d, ' % (clusterVec[3][0], clusterVec[3][1]))        
            fout.write('\n')
            
def saveTime(Xlabel, XRange, timeList, resultFilePath, titleStr):
    timeMeanList = []
    timeVarianceList = []
    for i in range(len(timeList)):
        timeMeanList.append(np.mean(timeList[i]))
        timeVarianceList.append(np.std(timeList[i]))
    with open(resultFilePath, 'a') as fout:
        fout.write('KRange/iterNumRange:' + repr(XRange) + '\n')
        fout.write('timeList:' + repr(timeList) + '\n')
        fout.write('timeMeanList:' + repr(timeMeanList) + '\n')
        fout.write('timeVarianceList:' + repr(timeVarianceList) + '\n')
        fout.write('\n#%s\n' % (titleStr)) 
        fout.write('\n#IterNum/K\tMean\tSD\n') 
        for j in range(len(timeMeanList)):
            fout.write('%d\t%.4f\t%.4f\n' % (XRange[j], timeMeanList[j], timeVarianceList[j]))
    py.plot(XRange, timeMeanList, 'b-*')
    py.errorbar(XRange, timeMeanList, yerr=timeVarianceList, fmt='b*')
    py.xlabel(Xlabel)
    py.ylabel('Time/sec')      
    py.title(titleStr)
    py.grid(True)
    py.show()

def drawPredK(dataset, resultFilePath, titleStr, Xlabel, XRange, KPredNumMeanList, KPredNumVarianceList):
    with open(resultFilePath, 'a') as fout:
        fout.write('KRange/iterNumRange:' + repr(XRange) + '\n')
        fout.write('KPredNumMeanList:' + repr(KPredNumMeanList) + '\n')
        fout.write('KPredNumVarianceList:' + repr(KPredNumVarianceList) + '\n')
        fout.write('\n#%s\n' % (titleStr)) 
        fout.write('\n#K\tMean\tSD\n') 
        for j in range(len(KPredNumMeanList)):
            fout.write('%.3f\t%.4f\t%.4f\n' % (XRange[j], KPredNumMeanList[j], KPredNumVarianceList[j]))
    
    Ylabel = 'The number of topics fround by MStream'
    py.figure()
    py.plot(XRange, KPredNumMeanList, 'bo')
    py.errorbar(XRange, KPredNumMeanList, yerr=KPredNumVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def evaluatePerSample(self, sampleNo):
        AMI = metrics.adjusted_mutual_info_score(self.labelsTrue, self.labelsPred)  
        NMI = metrics.normalized_mutual_info_score(self.labelsTrue, self.labelsPred)  
        MI = metrics.mutual_info_score(self.labelsTrue, self.labelsPred)  
        ARI = metrics.adjusted_rand_score(self.labelsTrue, self.labelsPred)  
        homogeneity = metrics.homogeneity_score(self.labelsTrue, self.labelsPred)  
        completeness = metrics.completeness_score(self.labelsTrue, self.labelsPred)  
        V = metrics.v_measure_score(self.labelsTrue, self.labelsPred)    
#        SC = metrics.silhouette_score(self.X, self.labelsPred, metric='sqeuclidean') #Silhouette Coefficient
        self.AMIList.append(AMI)
        self.NMIList.append(NMI)
        self.MIList.append(MI)
        self.ARIList.append(ARI)
        self.homogeneityList.append(homogeneity)
        self.completenessList.append(completeness)
        self.VList.append(V)

def MStreamBatchNum():
    K = 0
    # BatchNumRange = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    BatchNumRange = [1,6,11,16,21,31,41]
    BatchNumRangeStr = ''
    sampleNum = 1
    alpha = '40'
    beta = '0.008'
    iterNum = 2
    KThreshold = 0
    # batchNum = 16
    dataset = 'tweetsByTopics'
    datasetPath = './MStream/data/' + dataset
    inPath = './MStream/result_diffBatchNum/'
    resultFileName = 'MStreamNoiseKThreshold%dIterNumDataset%sK%dsampleNum%dalpha%sbeta%sIterNum%dBatchNum%s.txt' % (KThreshold,
                                  dataset, K, sampleNum, alpha, beta, iterNum, BatchNumRangeStr)
    resultFilePath = './MStream/result_diffBatchNum/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    noiseNumMeanList = []
    noiseNumVarianceList = []

    for BatchNum in BatchNumRange:
        KPredNumList = []
        noiseNumList = []
        for sampleNo in range(1, sampleNum+1):
            KPredNum_Batch = 0
            docRemainNum_Batch = 0
            docTotalNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for batch in range(1, BatchNum + 1):
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatchNum%dBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, BatchNum, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
                docRemainNum_Batch += docRemainNum
                docTotalNum_Batch += docTotalNum
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            KPredNumList.append(KPredNum_Batch)
            noiseNumList.append(docTotalNum_Batch - docRemainNum_Batch)
        KPredNumMeanList.append(np.mean(KPredNumList))
        noiseNumMeanList.append(np.mean(noiseNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        noiseNumVarianceList.append(np.std(noiseNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
    titleStr = 'BatchNum to MStream %s K%dSampleNum%dalpha%sbeta%s' % (dataset, K, sampleNum, alpha, beta)
    Xlabel = 'The number of BatchNum'
    MStreamEvaluation.drawEvaluationResult(BatchNumRange, Xlabel, titleStr)
    titlePredK = 'KPredNum%s KThreshold%d K%dSampleNum%dalpha%sbeta%s' % (
    dataset, KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, BatchNumRange, KPredNumMeanList, KPredNumVarianceList)
    titleRemainDoc = 'noiseNum%s KThres%d K%dS%dalpha%sbeta%s' % (dataset, KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titleRemainDoc, Xlabel, BatchNumRange, noiseNumMeanList, noiseNumVarianceList)

    timeFilePara = 'Time%sMStreamDiffBatchNumK%diterNum%dSampleNum%dalpha%sbeta%s' % (
        dataset, K, iterNum, sampleNum, alpha, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    timeList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                timeList.append([timeRun * 1.0 / 1000 for timeRun in timeObj['Times']])
                parameterList = timeObj['parameters']
            except:
                print(timeJson)
    timeSampleList = [[0 for j in range(sampleNum)] for i in range(len(parameterList))]
    for j in range(sampleNum):
        for i in range(len(parameterList)):
            timeSampleList[i][j] = timeList[j][i]

    for i in range(len(parameterList)):
        (timeMean, timeVariance) = MeanAndVar(timeSampleList[i])
        timeMeanList.append(timeMean)
        timeVarianceList.append(timeVariance)

    Xlabel = 'BatchNum'
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different BatchNum'
    py.figure()
    py.plot(parameterList, timeMeanList, 'b-o')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def MStreamMaxBatch():
    K = 0
    # MaxBatchRange = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    MaxBatchRange = [1,2,3,4,5,6,7,8]
    MaxBatchRangeStr = ''
    sampleNum = 1
    alpha = '40'
    beta = '0.008'
    iterNum = 2
    KThreshold = 0
    batchNum = 16
    dataset = 'tweetsByTopics'
    datasetPath = './MStream/data/' + dataset
    inPath = './MStream/result_diffMaxBatch/'
    resultFileName = 'MStreamNoiseKThreshold%dIterNumDataset%sK%dsampleNum%dalpha%sbeta%sIterNum%dMaxBatch%s.txt' % (KThreshold,
                                  dataset, K, sampleNum, alpha, beta, iterNum, MaxBatchRangeStr)
    resultFilePath = './MStream/result_diffMaxBatch/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    noiseNumMeanList = []
    noiseNumVarianceList = []

    for MaxBatch in MaxBatchRange:
        KPredNumList = []
        noiseNumList = []
        for sampleNo in range(1, sampleNum+1):
            KPredNum_Batch = 0
            docRemainNum_Batch = 0
            docTotalNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for batch in range(1, batchNum + 1):
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sMaxBatch%dBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, MaxBatch, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
                docRemainNum_Batch += docRemainNum
                docTotalNum_Batch += docTotalNum
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            KPredNumList.append(KPredNum_Batch)
            noiseNumList.append(docTotalNum_Batch - docRemainNum_Batch)
        KPredNumMeanList.append(np.mean(KPredNumList))
        noiseNumMeanList.append(np.mean(noiseNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        noiseNumVarianceList.append(np.std(noiseNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
    titleStr = 'MaxBatch to MStream %s K%dSampleNum%dalpha%sbeta%s' % (dataset, K, sampleNum, alpha, beta)
    Xlabel = 'The number of MaxBatch'
    MStreamEvaluation.drawEvaluationResult(MaxBatchRange, Xlabel, titleStr)
    titlePredK = 'KPredNum%s KThreshold%d K%dSampleNum%dalpha%sbeta%s' % (
    dataset, KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, MaxBatchRange, KPredNumMeanList, KPredNumVarianceList)
    titleRemainDoc = 'noiseNum%s KThres%d K%dS%dalpha%sbeta%s' % (dataset, KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titleRemainDoc, Xlabel, MaxBatchRange, noiseNumMeanList, noiseNumVarianceList)

    timeFilePara = 'Time%sMStreamDiffMaxBatchK%diterNum%dSampleNum%dalpha%sbeta%s' % (
        dataset, K, iterNum, sampleNum, alpha, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    timeList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                timeList.append([timeRun * 1.0 / 1000 for timeRun in timeObj['Times']])
                parameterList = timeObj['parameters']
            except:
                print(timeJson)
    timeSampleList = [[0 for j in range(sampleNum)] for i in range(len(parameterList))]
    for j in range(sampleNum):
        for i in range(len(parameterList)):
            timeSampleList[i][j] = timeList[j][i]

    for i in range(len(parameterList)):
        (timeMean, timeVariance) = MeanAndVar(timeSampleList[i])
        timeMeanList.append(timeMean)
        timeVarianceList.append(timeVariance)

    Xlabel = 'BatchNum'
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different BatchNum'
    py.figure()
    py.plot(parameterList, timeMeanList, 'b-o')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def MStreamIterNum():
    K = 0
    # iterNumRange = [0,1,2,3,4,5,6,7,8,9,10]
    iterNumRange = [0,1,2,3,4,5,6,7,8,9,10,20,25,30,35,40,45,50,55,60,65,70]
    iterNumRangeStr = ''
    sampleNum = 1
    alpha = '40'
    beta = '0.02'
    KThreshold = 0
    batchNum = 16
    dataset = 'newtweets'
    datasetPath = './MStream/data/' + dataset
    inPath = './MStream/result_diffIter/'
    resultFileName = 'MStreamNoiseKThreshold%dIterNumDataset%sK%dsampleNum%dalpha%sbeta%sIterNum%s.txt' % (KThreshold,
        dataset, K, sampleNum, alpha, beta,iterNumRangeStr)
    resultFilePath = './MStream/result_diffIter/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)
    
    KPredNumMeanList = []
    KPredNumVarianceList = []    
    noiseNumMeanList = []
    noiseNumVarianceList = []

    for iterNum in iterNumRange:
        KPredNumList = []
        noiseNumList = []
        for sampleNo in range(1, sampleNum+1):
            KPredNum_Batch = 0
            docRemainNum_Batch = 0
            docTotalNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for batch in range(1, batchNum + 1):
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
                docRemainNum_Batch += docRemainNum
                docTotalNum_Batch += docTotalNum
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            KPredNumList.append(KPredNum_Batch)
            noiseNumList.append(docTotalNum_Batch - docRemainNum_Batch)
        KPredNumMeanList.append(np.mean(KPredNumList))
        noiseNumMeanList.append(np.mean(noiseNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        noiseNumVarianceList.append(np.std(noiseNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
    titleStr = 'IterNum to MStream %s K%dSampleNum%dalpha%sbeta%s' % (dataset, K, sampleNum, alpha, beta)
    Xlabel = 'The number of iterations'
    MStreamEvaluation.drawEvaluationResult(iterNumRange, Xlabel, titleStr)

    titlePredK = 'KPredNum%s KThreshold%d K%dSampleNum%dalpha%sbeta%s' % (dataset,KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, iterNumRange, KPredNumMeanList, KPredNumVarianceList)
    
    titleRemainDoc = 'noiseNum%s KThres%d K%dS%dalpha%sbeta%s' % (dataset,KThreshold, K, sampleNum, alpha, beta)
    drawPredK(dataset, resultFilePath, titleRemainDoc, Xlabel, iterNumRange, noiseNumMeanList, noiseNumVarianceList)

    timeFilePara = 'Time%sMStreamDiffIterK%dSampleNum%dalpha%sbeta%s' % (
        dataset, K, sampleNum, alpha, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    timeList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:    
                timeObj = json.loads(timeJson)
                timeList.append([ timeRun*1.0/1000 for timeRun in timeObj['Times']])
                parameterList = timeObj['parameters']
            except:
                print(timeJson)
    timeSampleList = [ [0 for j in range(sampleNum)] for i in range(len(parameterList))]
    for j in range(sampleNum):
        for i in range(len(parameterList)):
            timeSampleList[i][j] = timeList[j][i]
    
    for i in range(len(parameterList)):
        (timeMean, timeVariance) = MeanAndVar(timeSampleList[i])
        timeMeanList.append(timeMean)
        timeVarianceList.append(timeVariance)
    
    Xlabel = 'Iterations'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different iterations'
    py.figure()
    py.plot(parameterList, timeMeanList, 'b-o')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def MStreamBeta():
    K = 0
    docNum = 1
    # betaRange = ['0.005','0.006','0.007', '0.008', '0.009', '0.01']
    # betaRange_float = [0.005,0.006,0.007,0.008,0.009,0.01]
    # betaRange = [str(round(i * 0.01 * docNum, 2)) for i in range(1, 10, 1)] + \
    #              [str(round(j * 0.1 * docNum, 1)) for j in range(1, 10)] + [str(1.0)]
    # betaRange_float = [round(i * 0.01 * docNum, 2) for i in range(1, 10, 1)] + \
    #                    [round(j * 0.1 * docNum, 1) for j in range(1, 10)] + [1.0]
    # betaRange = ['0.02']
    # betaRange_float = [0.02]
    betaRange = ['0.014','0.016','0.018','0.02','0.022','0.024']
    betaRange_float = [0.014,0.016,0.018,0.02,0.022,0.024]
    # betaRange = ['0.016', '0.018', '0.02', '0.022', '0.024', '0.026', '0.028', '0.03', '0.032', '0.034', '0.04', '0.05', '0.06', '0.07']
    # betaRange_float = [0.016, 0.018, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.032, 0.034, 0.04, 0.05, 0.06, 0.07]
    # betaRange = ['0.02', '0.03', '0.04', '0.05','0.06', '0.07', '0.08', '0.09', '0.1']
    # betaRange_float = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    print("betaRange is \n", betaRange)
    # betaRangeStr = '0.01range(1, 11, 1)0.1range(1, 11, 1)'
    betaRangeStr = ''
    sampleNum = 1
    iterNum = 2
    alpha = "0.02"
    # alpha_float = 1800.0
    KThreshold = 0
    batchNum = 16
    dataset = 'newtweets'
    # dataset = 'tweetsByTopics'
    # dataset = 'TweetsEvents22'
    # dataset = 'TweetsEvents22ByTopics'
    datasetPath = './MStream/data/' + dataset
    inPath = './MStream/result_improve/'
    resultFileName = 'MStreamKThreshold%dBetaDataset%salpha%sK%dIterNum%dBetaRange%ssampleNum%d.txt' % (KThreshold, dataset,alpha, K,iterNum, betaRangeStr, sampleNum)
    resultFilePath = './MStream/result_improve/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    for beta in betaRange:
        print("beta: ", beta)
        KPredNumList = []
        for sampleNo in range(1, sampleNum + 1):
            KPredNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for batch in range(1, batchNum + 1):
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    '''
    KPredNumMeanList = []
    KPredNumVarianceList = []    
    for beta in betaRange:
        dirName = '%sK%diterNum%dSampleNum%dalpha%.3fbeta%.3f/' % (dataset, K, iterNum, sampleNum, alpha, beta)
        inDir = inPath + dirName
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamLabels(inFile, datasetPath)
            MStreamEvaluation.evaluatePerSample(sampleNo)
            sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
            (KPredNum,docRemainNum,docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
            KPredNumList.append(KPredNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
        MStreamEvaluation.evaluateAllSamples(iterNum)
    '''
    titleStr = 'beta to MStream on %s K%dSampleNum%dalpha%sIter%d' % (dataset, K, sampleNum, alpha,  iterNum)
    Xlabel = 'beta'
    MStreamEvaluation.drawEvaluationResult(betaRange_float, Xlabel, titleStr)
    titlePredK = '%s KThreshold%d K%dSampleNum%dalpha%sIter%d' % (dataset, KThreshold, K, sampleNum, alpha,  iterNum)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, betaRange_float, KPredNumMeanList, KPredNumVarianceList)

    timeFilePara = 'Time%sMStreamDiffBetaK%diterNum%dSampleNum%dalpha%s' % (dataset, K, iterNum, sampleNum, alpha)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                parameterList.append(timeObj['parameter'])
                (timeMean, timeVariance) = MeanAndVar([ timeRun for timeRun in timeObj['Time']])
                timeMeanList.append(timeMean)
                timeVarianceList.append(timeVariance)
            except:
                print(timeJson)
    Xlabel = 'Beta'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different betas'
    py.figure()
    py.plot(parameterList, timeMeanList, 'bo')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def MStreamAlpha():
    K = 0
    docNum = 1
    alphaRange = ['0.005', '0.006', '0.007', '0.008', '0.009', '0.01', '0.02', '0.03', '0.04', '0.05', '0.06']
    alphaRange_float = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    print("alphaRange is \n", alphaRange)
    alphaRangeStr = ''
    sampleNum = 1
    iterNum = 2
    beta = "0.02"
    beta_float = 0.02
    KThreshold = 0
    batchNum = 16
    dataset = 'newtweets'
    # dataset = 'tweetsByTopics'
    # dataset = 'TweetsEvents22'
    # dataset = 'TweetsEvents22ByTopics'
    datasetPath = './MStream/data/' + dataset
    # inPath = './MStream/result/'
    inPath = './MStream/result_improve/'
    resultFileName = 'MStreamAlphaKThreshold%dDataset%sbeta%sK%dIterNum%dAlphaRange%ssampleNum%d.txt' % (KThreshold, dataset, beta, K,iterNum, alphaRangeStr, sampleNum)
    # resultFilePath = './MStream/result/' + resultFileName
    resultFilePath = './MStream/result_improve/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    for alpha in alphaRange:
        print("alpha is ", alpha)
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            KPredNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for batch in range(1, batchNum+1):
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    '''
    for alpha in alphaRange:
        for batch in range(batchNum):
            dirName = '%sK%diterNum%dSampleNum%dalpha%.1fbeta%.2fBatch%d/' % (dataset, K, iterNum, sampleNum, alpha, beta, batch)
            inDir = inPath + dirName
            KPredNumList = []
            for sampleNo in range(1, sampleNum + 1):
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamLabels(inFile, datasetPath)
                MStreamEvaluation.evaluatePerSample(sampleNo)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNumList.append(KPredNum)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    '''
    titleStr = 'alpha to MStream on  %s K%dSampleNum%dbeta%sIter%d' % (dataset, K, sampleNum, beta,  iterNum)
    Xlabel = 'alpha'
    titlePredK = '%s KThreshold%d K%dSampleNum%dbeta%sIter%d' % (dataset, KThreshold, K, sampleNum, beta,  iterNum)

    MStreamEvaluation.drawEvaluationResult(alphaRange_float, Xlabel, titleStr)
    # MStreamEvaluation.drawEvaluationResult(alphaRange, Xlabel, titleStr)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, alphaRange_float, KPredNumMeanList, KPredNumVarianceList)

    timeFilePara = 'Time%sMStreamDiffAlphaK%diterNum%dSampleNum%dbeta%s' % (dataset, K, iterNum, sampleNum, beta)
    timeFilePath = inPath + timeFilePara + '.txt'
    parameterList = []
    timeMeanList = []
    timeVarianceList = []
    with codecs.open(timeFilePath, 'r', 'utf-8') as fin:
        for timeJson in fin:
            try:
                timeObj = json.loads(timeJson)
                parameterList.append(timeObj['parameter'])
                (timeMean, timeVariance) = MeanAndVar([ timeRun for timeRun in timeObj['Time']])
                timeMeanList.append(timeMean)
                timeVarianceList.append(timeVariance)
            except:
                print(timeJson)
    Xlabel = 'Alpha'            
    Ylabel = 'Time'
    titleStr = 'MStream Running time with different alphas'
    py.figure()
    print(timeMeanList)
    py.plot(parameterList, timeMeanList, 'bo')
    py.errorbar(parameterList, timeMeanList, yerr=timeVarianceList, fmt='bo')
    py.xlabel(Xlabel)
    py.ylabel(Ylabel)
    py.title(titleStr)
    py.grid(True)
    py.show()

def MStream_Batch():
    BatchRange_float = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    K = 0
    # docNum = 1
    sampleNum = 1
    iterNum = 2
    alpha = '0.02'
    # alpha_float = 300
    beta = "0.02"
    # beta_float = 0.02
    KThreshold = 0
    batchNum = 16
    dataset = 'newtweets'
    # dataset = 'tweetsByTopics'
    # dataset = 'TweetsEvents22'
    # dataset = 'TweetsEvents22ByTopics'
    datasetPath = './MStream/data/' + dataset
    # inPath = './MStream/result/'
    inPath = './MStream/result_improve/'
    resultFileName = 'Batch_MStreamKThreshold%dDataset%salpha%sbeta%sK%dIterNum%dsampleNum%d.txt' % \
                     (KThreshold, dataset, alpha, beta, K, iterNum, sampleNum)
    # resultFilePath = './MStream/result/' + resultFileName
    resultFilePath = './MStream/result_improve/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    for batch in BatchRange_float:
        print("batch is ", batch)
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            MStreamEvaluation.labelsPred = {}
            dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatch%d/' % (dataset, K, iterNum, sampleNum, alpha, beta, batch)
            inDir = inPath + dirName
            fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
            inFile = inDir + fileName
            MStreamEvaluation.getMStreamPredLabels(inFile)
            sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
            (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
            MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha +
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum)
            MStreamEvaluation.evaluatePerSample(sampleNo)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    titleStr = 'MStream EvaluationOfEachBatch %s K%dSampleNum%dalpha%sbeta%sIter%d' % (dataset, K, sampleNum, alpha, beta, iterNum)
    Xlabel = 'batch'
    titlePredK = '%s KThreshold%d K%dSampleNum%dalpha%sbeta%sIter%d' % (dataset, KThreshold, K, sampleNum, alpha, beta, iterNum)
    MStreamEvaluation.drawEvaluationResult(BatchRange_float, Xlabel, titleStr)
    # MStreamEvaluation.drawEvaluationResult(alphaRange, Xlabel, titleStr)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, BatchRange_float, KPredNumMeanList, KPredNumVarianceList)

def MStream_AlphaBatch():
    BatchRange_float = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    K = 0
    docNum = 1
    sampleNum = 5
    iterNum = 8
    alphaRange = ['10', '20', '30', '40', '50', '60', '70']
    alphaRange_float = [10,20,30,40,50,60,70]
    alphaRange_str = '10-70'
    beta = "0.02"
    beta_float = 0.02
    KThreshold = 0
    batchNum = 16
    dataset = 'newtweets'
    datasetPath = './MStream/data/' + dataset
    # inPath = './MStream/result/'
    inPath = './MStream/result_improve/'
    resultFileName = 'Batch_Alpha%s_MStreamKThreshold%dDataset%sbeta%sK%dIterNum%dsampleNum%d.txt' % \
                     (alphaRange_str, KThreshold, dataset, beta, K, iterNum, sampleNum)
    # resultFilePath = './MStream/result/' + resultFileName
    resultFilePath = './MStream/result_improve/' + resultFileName
    MStreamEvaluation = ClusterEvaluation(resultFilePath)

    KPredNumMeanList = []
    KPredNumVarianceList = []
    for batch in BatchRange_float:
        print("batch is ", batch)
        KPredNumList = []
        for sampleNo in range(1, sampleNum+1):
            KPredNum_Batch = 0
            MStreamEvaluation.labelsPred = {}
            for alpha in alphaRange:
                dirName = '%sK%diterNum%dSampleNum%dalpha%sbeta%sBatch%d/' % \
                          (dataset, K, iterNum, sampleNum, alpha, beta, batch)
                inDir = inPath + dirName
                fileName = '%sSampleNo%dClusteringResult.txt' % (dataset, sampleNo)
                inFile = inDir + fileName
                MStreamEvaluation.getMStreamPredLabels(inFile)
                sizeFile = inDir + '%sSampleNo%dSizeOfEachCluster.txt' % (dataset, sampleNo)
                (KPredNum, docRemainNum, docTotalNum) = MStreamEvaluation.getPredNumThreshold(sizeFile, KThreshold)
                KPredNum_Batch += KPredNum
                MStreamEvaluation.getMStreamTrueLabels(inPath + dataset + "K" + str(K) + "iterNum" + str(iterNum) + \
                                                   "SampleNo" + str(sampleNo) + "alpha" + alpha + \
                                                   "beta" + beta, datasetPath)
            KPredNumList.append(KPredNum_Batch)
            MStreamEvaluation.evaluatePerSample(sampleNo)
        MStreamEvaluation.evaluateAllSamples(iterNum)
        KPredNumMeanList.append(np.mean(KPredNumList))
        KPredNumVarianceList.append(np.std(KPredNumList))
    titleStr = 'MStreamBatch Alpha%s %s K%dSampleNum%dbeta%sIter%d' % (alphaRange_str, dataset, K, sampleNum, beta, iterNum)
    Xlabel = 'batch'
    titlePredK = 'Alpha%s %s KThreshold%d K%dSampleNum%dbeta%sIter%d' % (alphaRange_str, dataset, KThreshold, K, sampleNum, beta, iterNum)
    MStreamEvaluation.drawEvaluationResult(BatchRange_float, Xlabel, titleStr)
    # MStreamEvaluation.drawEvaluationResult(alphaRange, Xlabel, titleStr)
    drawPredK(dataset, resultFilePath, titlePredK, Xlabel, BatchRange_float, KPredNumMeanList, KPredNumVarianceList)

if __name__ == '__main__':
    # MStreamBatchNum()
    # MStreamMaxBatch()
    # MStreamIterNum()
    MStreamBeta()
    # MStreamAlpha()
    # MStream_Batch() # Get the evaluation for each batch
    # MStream_AlphaBatch() # Get the evaluation for each batch according the average of different alpha
