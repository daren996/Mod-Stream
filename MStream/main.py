from MStream import MStream
import json
import time

dataDir = "data/"
outputPath = "result/"
outputPath_improve = "result_improve/"

# dataset = "TREC"
dataset = "TREC-T"
# dataset = "Tweets"
# dataset = "Tweets-T"

timefil = "timefil"
MaxBatch = 1
# docNum = 2300
alpha = 0.02
K = 0 # Number of clusters
beta = 0.02
iterNum = 10
sampleNum = 10
wordsInTopicNum = 5

def runMStreamSimple(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    mstream.getDocuments()
    for sampleNo in range(1, sampleNum+1):
        print("SampleNo:"+str(sampleNo))
        mstream.runMStream(sampleNo)

def runMStreamSimple_improve(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    mstream.getDocuments()
    for sampleNo in range(1, sampleNum+1):
        print("SampleNo:"+str(sampleNo))
        mstream.runMStream_improve(sampleNo)

def runWithAlphaScale(beta, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum, docNum):
    parameters = []
    timeArrayOfParas = []
    p = 100
    while p <= 300:
        alpha = docNum * p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 10
    fileParameters = "MStreamDiffAlpha" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + \
                     str(sampleNum) + "beta" + str(round(beta, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithAlphaScale_improve(beta, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    p = 0.01
    while p <= 0.05:
        alpha = p
        parameters.append(p)
        print("alpha:", alpha, "\tp:", p)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo)
            startTime = time.time()
            mstream.runMStream_improve(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        p += 0.01
    fileParameters = "MStreamDiffAlpha" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + \
                     str(sampleNum) + "beta" + str(round(beta, 3))
    outTimePath = outputPath_improve + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBetas(alpha, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    beta = 0.008
    while beta <= 0.0101:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.001
    beta = 0.01
    while beta <= 0.051:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.01
    fileParameters = "MStreamDiffBeta" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBetas_improve(alpha, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    beta = 0.014
    while beta <= 0.026:
        parameters.append(beta)
        print("beta:", beta)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream_improve(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        beta += 0.002
    fileParameters = "MStreamDiffBeta" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3))
    outTimePath = outputPath_improve + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithNiters(K, MaxBatch, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    iterNum = 0
    while iterNum <= 10:
        parameters.append(iterNum)
        print("iterNum:", iterNum)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        iterNum += 1
    iterNum = 20
    while iterNum <= 100:
        parameters.append(iterNum)
        print("iterNum:", iterNum)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream(sampleNo)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        iterNum += 5
    fileParameters = "MStreamDiffIter" + "K" + str(K) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithBatchNum(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    BatchNum = 26
    while BatchNum <= 100:
        parameters.append(BatchNum)
        print("BatchNum:", BatchNum)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream_withBatchNum(sampleNo, BatchNum)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        BatchNum += 10
    fileParameters = "MStreamDiffBatchNum" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()

def runWithMaxBatch(K,  alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
    parameters = []
    timeArrayOfParas = []
    MaxBatch = 6
    while MaxBatch <= 16:
        parameters.append(MaxBatch)
        print("MaxBatch:", MaxBatch)
        mstream = MStream(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
        mstream.getDocuments()
        timeArray = []
        for sampleNo in range(1, sampleNum + 1):
            print("SampleNo:", sampleNo, end=' ')
            startTime = time.time()
            mstream.runMStream_withMaxBatch(sampleNo, MaxBatch)
            endTime = time.time()
            timeArray.append(int(endTime - startTime))
        timeArrayOfParas.append(timeArray)
        MaxBatch += 1
    fileParameters = "MStreamDiffMaxBatch" + "K" + str(K) + "iterNum" + str(iterNum) + "SampleNum" + str(sampleNum) + \
                     "alpha" + str(round(alpha, 3)) + "beta" + str(round(beta, 3))
    outTimePath = outputPath + "Time" + dataset + fileParameters + ".txt"
    writer = open(outTimePath, 'w')
    parasNum = parameters.__len__()
    for i in range(parasNum):
        temp_obj = {}
        temp_obj['parameter'] = parameters[i]
        temp_obj['Time'] = timeArrayOfParas[i]
        temp_json = json.dumps(temp_obj)
        writer.write(temp_json)
        writer.write('\n')
    writer.close()


if __name__ == '__main__':
    # runMStreamSimple(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    runMStreamSimple_improve(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithAlphaScale(beta, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum, docNum)
    # runWithAlphaScale_improve(beta, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBetas(alpha, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBetas_improve(alpha, K, MaxBatch, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithNiters(K, MaxBatch, alpha, beta, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithBatchNum(K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)
    # runWithMaxBatch(K, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum)