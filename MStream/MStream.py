from DocumentSet import DocumentSet
from Model import Model

class MStream:
    dataDir = "data/"
    outputPath = "result/"
    outputPath_improve = "result_improve/"

    def __init__(self, K, MaxBatch, alpha, beta, iterNum, sampleNum, dataset, timefil, wordsInTopicNum):
        self.K = K
        self.MaxBatch = MaxBatch
        self.alpha = alpha
        self.beta = beta
        self.iterNum = iterNum
        self.dataset = dataset
        self.timefil = timefil
        self.sampleNum = sampleNum
        self.wordsInTopicNum = wordsInTopicNum
        self.wordList = []
        self.wordToIdMap = {}

    def getDocuments(self):
        self.documentSet = DocumentSet(self.dataDir + self.dataset, self.wordToIdMap, self.wordList)
        self.V = self.wordToIdMap.__len__()

    def runMStream(self, sampleNo):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3))
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run(self.documentSet, self.outputPath, self.wordList)

    def runMStream_withBatchNum(self, sampleNo, AllBatchNum):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3))
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run_withAllBatchNum(self.documentSet, self.outputPath, self.wordList, AllBatchNum)

    def runMStream_withMaxBatch(self, sampleNo, MaxBatch):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3)) + "MaxBatch" + str(MaxBatch)
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run(self.documentSet, self.outputPath, self.wordList)

    def runMStream_improve(self, sampleNo):
        ParametersStr = "K" + str(self.K) + "iterNum" + str(self.iterNum) + \
                        "SampleNum" + str(self.sampleNum) + "alpha" + str(round(self.alpha, 3)) + \
                        "beta" + str(round(self.beta, 3))
        model = Model(self.K, self.MaxBatch, self.V, self.iterNum, self.alpha, self.beta, self.dataset,
                      ParametersStr, sampleNo, self.wordsInTopicNum, self.dataDir + self.timefil)
        model.run_improve(self.documentSet, self.outputPath_improve, self.wordList)

