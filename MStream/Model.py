import random
import os
import time
import json
import copy

AllBatchNum = 16
threshold_fix = 1.1
threshold_init = 0

class Model:

    def __init__(self, K, Max_Batch, V, iterNum,alpha0, beta, dataset, ParametersStr, sampleNo, wordsInTopicNum, timefil):
        self.dataset = dataset
        self.ParametersStr = ParametersStr
        self.alpha0 = alpha0
        self.beta = beta
        self.K = K
        self.Kin = K
        self.V = V
        self.iterNum = iterNum
        self.beta0 = float(V) * float(beta)
        self.sampleNo = sampleNo
        self.wordsInTopicNum = copy.deepcopy(wordsInTopicNum)
        self.Max_Batch = Max_Batch  # Max number of batches we will consider
        self.phi_zv = []

        self.batchNum2tweetID = {} # batch num to tweet id
        self.batchNum = 1 # current batch number
        self.readTimeFil(timefil)

    def readTimeFil(self, timefil):
        try:
            with open(timefil) as timef:
                for line in timef:
                    buff = line.strip().split(' ')
                    if buff == ['']:
                        break
                    self.batchNum2tweetID[self.batchNum] = int(buff[1])
                    self.batchNum += 1
            self.batchNum = 1
        except:
            print("No timefil!")
        # print("There are", self.batchNum2tweetID.__len__(), "time points.\n\t", self.batchNum2tweetID)

    def getAveBatch(self, documentSet, AllBatchNum):
        self.batchNum2tweetID.clear()
        temp = self.D_All / AllBatchNum
        _ = 0
        count = 1
        for d in range(self.D_All):
            if _ < temp:
                _ += 1
                continue
            else:
                document = documentSet.documents[d]
                documentID = document.documentID
                self.batchNum2tweetID[count] = documentID
                count += 1
                _ = 0
        self.batchNum2tweetID[count] = -1

    def run_withAllBatchNum(self, documentSet, outputPath, wordList, AllBatchNum):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch

        # Get batchNum2tweetID by AllBatchNum
        self.AllBatchNum = AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.checkEmpty(cluster)
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output_withAllBatchNum(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")

    def run(self, documentSet, outputPath, wordList):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.checkEmpty(cluster)
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.intialize(documentSet)
                self.gibbsSampling(documentSet)
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")

    def intialize(self, documentSet):
        for d in range(self.currentDoc, self.D_All):
            documentID = documentSet.documents[d].documentID
            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        print("\t" + str(self.D) + " documents will be analyze")
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            if documentID != self.batchNum2tweetID[self.batchNum]:
                cluster = self.sampleCluster(d, document)  # Get initial cluster of each document
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

    def gibbsSampling(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i)
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                cluster = self.z[documentID]
                self.m_z[cluster] -= 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    self.n_zv[cluster][wordNo] -= wordFre
                    self.n_z[cluster] -= wordFre
                self.checkEmpty(cluster)
                cluster = self.sampleCluster(d, document)
                self.z[documentID] = cluster
                if cluster not in self.m_z:
                    self.m_z[cluster] = 0
                self.m_z[cluster] += 1
                for w in range(document.wordNum):
                    wordNo = document.wordIdArray[w]
                    wordFre = document.wordFreArray[w]
                    if cluster not in self.n_zv:
                        self.n_zv[cluster] = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    self.n_zv[cluster][wordNo] += wordFre
                    self.n_z[cluster] += wordFre

    def sampleCluster(self, d, document):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha0)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha0 / (self.D - 1 + self.alpha0)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        for k in range(1, self.K+1):
            prob[k] += prob[k-1]
        thred = random.random() * prob[self.K]
        kChoosed = 0
        while kChoosed < self.K+1:
            if thred < prob[kChoosed]:
                break
            kChoosed += 1
        if kChoosed == self.K:
            self.K += 1
            self.K_current += 1
        return kChoosed

    '''
    When the probability that this document belongs to a cluster is greater than threshold_init, this document can be initialized.
    When the probability that this document belongs to a cluster is greater than threshold_fix, this document can be fixed.
    Fixed means that its class will not change again.
    Assigned documents at maximum probability on the last iteration and fixing.
    # Cluster with only one document is outlier. Outlier means that this class is no longer considered when iterating.
    
    At initialization, V_current has another choices:
    set V_current is the number of words analyzed in the current batch and the number of words in previous batches,
        which means beta0s of each document are different at initialization.
 
    We use self.hasIntialized{} to represent the situation of each document.
    0 means that it wasn't initialized
    1 means that documents have been initialized but wasn't fixed
    2 means that documents have been fixed
    
    Before we process each batch, alpha will be recomputed by the function: alpha = alpha0 * docStored
    '''
    def run_improve(self, documentSet, outputPath, wordList):
        self.D_All = documentSet.D  # The whole number of documents
        self.z = {}  # Cluster assignments of each document                 (documentID -> clusterID)
        self.m_z = {}  # The number of documents in cluster z               (clusterID -> number of documents)
        self.n_z = {}  # The number of words in cluster z                   (clusterID -> number of words)
        self.n_zv = {}  # The number of occurrences of word v in cluster z  (n_zv[clusterID][wordID] = number)
        self.currentDoc = 0  # Store start point of next batch
        self.startDoc = 0  # Store start point of this batch
        self.D = 0  # The number of documents currently
        self.K_current = copy.deepcopy(self.K) # the number of cluster containing documents currently
        self.BatchSet = {} # Store information of each batch
        self.word_current = {} # Store word-IDs' list of each batch

        # Get batchNum2tweetID by AllBatchNum
        self.getAveBatch(documentSet, AllBatchNum)
        print("batchNum2tweetID is ", self.batchNum2tweetID)

        while self.currentDoc < self.D_All:
            print("Batch", self.batchNum)
            if self.batchNum not in self.batchNum2tweetID:
                break
            if self.batchNum <= self.Max_Batch:
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.hasIntialized = {} # represents the situation of each document
                self.intialize_improve(documentSet)
                self.gibbsSampling_improve(documentSet)
                self.hasIntialized.clear()
            else:
                # remove influence of batch earlier than Max_Batch
                self.D -= self.BatchSet[self.batchNum - self.Max_Batch]['D']
                for cluster in self.m_z:
                    if cluster in self.BatchSet[self.batchNum - self.Max_Batch]['m_z']:
                        self.m_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['m_z'][cluster]
                        self.checkEmpty(cluster)
                        self.n_z[cluster] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_z'][cluster]
                        for word in self.n_zv[cluster]:
                            if word in self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster]:
                                self.n_zv[cluster][word] -= self.BatchSet[self.batchNum - self.Max_Batch]['n_zv'][cluster][word]
                self.BatchSet.pop(self.batchNum - self.Max_Batch)
                self.BatchSet[self.batchNum] = {}
                self.BatchSet[self.batchNum]['D'] = copy.deepcopy(self.D)
                self.BatchSet[self.batchNum]['z'] = copy.deepcopy(self.z)
                self.BatchSet[self.batchNum]['m_z'] = copy.deepcopy(self.m_z)
                self.BatchSet[self.batchNum]['n_z'] = copy.deepcopy(self.n_z)
                self.BatchSet[self.batchNum]['n_zv'] = copy.deepcopy(self.n_zv)
                self.hasIntialized = {}
                self.intialize_improve(documentSet)
                self.gibbsSampling_improve(documentSet)
                self.hasIntialized.clear()
            # get influence of only the current batch (remove other influence)
            self.BatchSet[self.batchNum-1]['D'] = self.D - self.BatchSet[self.batchNum-1]['D']
            for cluster in self.m_z:
                if cluster not in self.BatchSet[self.batchNum - 1]['m_z']:
                    self.BatchSet[self.batchNum - 1]['m_z'][cluster] = 0
                if cluster not in self.BatchSet[self.batchNum - 1]['n_z']:
                    self.BatchSet[self.batchNum - 1]['n_z'][cluster] = 0
                self.BatchSet[self.batchNum - 1]['m_z'][cluster] = self.m_z[cluster] - self.BatchSet[self.batchNum - 1]['m_z'][cluster]
                self.BatchSet[self.batchNum - 1]['n_z'][cluster] = self.n_z[cluster] - self.BatchSet[self.batchNum - 1]['n_z'][cluster]
                if cluster not in self.BatchSet[self.batchNum - 1]['n_zv']:
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster] = {}
                for word in self.n_zv[cluster]:
                    if word not in self.BatchSet[self.batchNum - 1]['n_zv'][cluster]:
                        self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = 0
                    self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word] = self.n_zv[cluster][word] - self.BatchSet[self.batchNum - 1]['n_zv'][cluster][word]
            print("\tGibbs sampling successful! Start to saving results.")
            self.output(documentSet, outputPath, wordList, self.batchNum - 1)
            print("\tSaving successful!")

    # beta0 for every batch
    def getBeta0_improve(self):
        Words = []
        if self.batchNum < 5:
            for i in range(1, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        if self.batchNum >= 5:
            for i in range(self.batchNum - 4, self.batchNum + 1):
                Words = list(set(Words + self.word_current[i]))
        return (float(len(list(set(Words)))) * float(self.beta))

    def intialize_improve(self, documentSet):
        self.word_current[self.batchNum] = []
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID
            # This method is getting beta0 at the beginning of initialization considering the whole words in current batch
            # for w in range(document.wordNum):
            #     wordNo = document.wordIdArray[w]
            #     if wordNo not in self.word_current[self.batchNum]:
            #         self.word_current[self.batchNum].append(wordNo)
            if documentID != self.batchNum2tweetID[self.batchNum]:
                self.D += 1
            else:
                break
        self.beta0 = self.getBeta0_improve()
        self.alpha = self.alpha0 * self.D
        print("\t" + str(self.D) + " documents will be analyze. alpha is" + " %.2f." % self.alpha + "\n\tInitialization.", end='\t')
        for d in range(self.currentDoc, self.D_All):
            document = documentSet.documents[d]
            documentID = document.documentID

            # This method is getting beta0 before each document is initialized
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                if wordNo not in self.word_current[self.batchNum]:
                    self.word_current[self.batchNum].append(wordNo)
            self.beta0 = self.getBeta0_improve()
            if self.beta0 <= 0:
                print("Wrong V!")
                exit(-1)

            if documentID != self.batchNum2tweetID[self.batchNum]:
                if documentID not in self.hasIntialized:
                    self.hasIntialized[documentID] = 0 # 0 means that it wasn't initialized
                if self.iterNum != 0:
                    cluster = self.sampleCluster_init_improve(d, document, documentID)
                if self.iterNum == 0:
                    cluster = self.sampleCluster_first_improve(d, document, documentID, 1)
                # 1 means that documents have been initialized but wasn't fixed
                # 2 means that documents have been fixed
                if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                    self.z[documentID] = cluster
                    if cluster not in self.m_z:
                        self.m_z[cluster] = 0
                    self.m_z[cluster] += 1
                    for w in range(document.wordNum):
                        wordNo = document.wordIdArray[w]
                        wordFre = document.wordFreArray[w]
                        if cluster not in self.n_zv:
                            self.n_zv[cluster] = {}
                        if wordNo not in self.n_zv[cluster]:
                            self.n_zv[cluster][wordNo] = 0
                        self.n_zv[cluster][wordNo] += wordFre
                        if cluster not in self.n_z:
                            self.n_z[cluster] = 0
                        self.n_z[cluster] += wordFre
                if d == self.D_All - 1:
                    self.startDoc = self.currentDoc
                    self.currentDoc = self.D_All
                    self.batchNum += 1
            else:
                self.startDoc = self.currentDoc
                self.currentDoc = d
                self.batchNum += 1
                break

        # Initialize documents which weren't initialized.
        notInitialized = 0
        for d in range(self.startDoc, self.currentDoc):
            document = documentSet.documents[d]
            documentID = document.documentID
            if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                continue
            if self.hasIntialized[documentID] == 0:
                notInitialized += 1
                cluster = self.sampleCluster_first_improve(d, document, documentID, 1)
                if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                    self.z[documentID] = cluster
                    if cluster not in self.m_z:
                        self.m_z[cluster] = 0
                    self.m_z[cluster] += 1
                    for w in range(document.wordNum):
                        wordNo = document.wordIdArray[w]
                        wordFre = document.wordFreArray[w]
                        if cluster not in self.n_zv:
                            self.n_zv[cluster] = {}
                        if wordNo not in self.n_zv[cluster]:
                            self.n_zv[cluster][wordNo] = 0
                        self.n_zv[cluster][wordNo] += wordFre
                        if cluster not in self.n_z:
                            self.n_z[cluster] = 0
                        self.n_z[cluster] += wordFre
        print("\t\t", notInitialized, "documents weren't initialized still.")

    def gibbsSampling_improve(self, documentSet):
        for i in range(self.iterNum):
            print("\titer is ", i+1, end='\t')
            notInitialized = 0 # number of documents not initialized
            notfixed = 0 # number of documents initialized but not fixed
            hasfixed = 0 # number of documents have been fixed
            for d in range(self.startDoc, self.currentDoc):
                document = documentSet.documents[d]
                documentID = document.documentID
                if self.hasIntialized[documentID] == 2: # documents fixed
                    hasfixed += 1
                    continue
                if self.hasIntialized[documentID] == 0:
                    notInitialized += 1
                    notfixed += 1
                    if i == 0: # if first iteration
                        cluster = self.sampleCluster_first_improve(d, document, documentID, 0)
                    elif i != 0 and i != self.iterNum - 1: # if not first iteration and not last iteration
                        cluster = self.sampleCluster_iter_improve(d, document, documentID, 0)
                    elif i == self.iterNum - 1: # if last iteration
                        cluster = self.sampleCluster_iter_improve(d, document, documentID, 1)
                    if self.hasIntialized[documentID] == 1 or self.hasIntialized[documentID] == 2:
                        self.z[documentID] = cluster
                        if cluster not in self.m_z:
                            self.m_z[cluster] = 0
                        self.m_z[cluster] += 1
                        for w in range(document.wordNum):
                            wordNo = document.wordIdArray[w]
                            wordFre = document.wordFreArray[w]
                            if cluster not in self.n_zv:
                                self.n_zv[cluster] = {}
                            if wordNo not in self.n_zv[cluster]:
                                self.n_zv[cluster][wordNo] = 0
                            self.n_zv[cluster][wordNo] += wordFre
                            if cluster not in self.n_z:
                                self.n_z[cluster] = 0
                            self.n_z[cluster] += wordFre
                    continue
                if self.hasIntialized[documentID] == 1:
                    notfixed += 1
                    cluster = self.z[documentID]
                    self.m_z[cluster] -= 1
                    for w in range(document.wordNum):
                        wordNo = document.wordIdArray[w]
                        wordFre = document.wordFreArray[w]
                        self.n_zv[cluster][wordNo] -= wordFre
                        self.n_z[cluster] -= wordFre
                    self.checkEmpty(cluster)
                    if i == 0: # if first iteration
                        cluster = self.sampleCluster_first_improve(d, document, documentID, 0)
                    elif i != 0 and i != self.iterNum - 1: # if not first iteration and not last iteration
                        cluster = self.sampleCluster_iter_improve(d, document, documentID, 0)
                    elif i == self.iterNum - 1: # if last iteration
                        cluster = self.sampleCluster_iter_improve(d, document, documentID, 1)
                    self.z[documentID] = cluster
                    if cluster not in self.m_z:
                        self.m_z[cluster] = 0
                    self.m_z[cluster] += 1
                    for w in range(document.wordNum):
                        wordNo = document.wordIdArray[w]
                        wordFre = document.wordFreArray[w]
                        if cluster not in self.n_zv:
                            self.n_zv[cluster] = {}
                        if wordNo not in self.n_zv[cluster]:
                            self.n_zv[cluster][wordNo] = 0
                        if cluster not in self.n_z:
                            self.n_z[cluster] = 0
                        self.n_zv[cluster][wordNo] += wordFre
                        self.n_z[cluster] += wordFre
            print("\t\t", notInitialized, "documents weren't initialized still.", notfixed, "documents weren't fixed still.")
            if notfixed == 0:
                print("All document has been initialized! Stop iteration.")
                return
        return

    # At initialzation iteration, a document may be initialized or not.
    def sampleCluster_init_improve(self, d, document, documentID):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities
        for k in range(self.K+1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k]/allProb)
        for k in range(self.K + 1):
            if prob_normalized[k] > threshold_fix:
                self.hasIntialized[documentID] = 2
                break
            if prob_normalized[k] > threshold_init:
                self.hasIntialized[documentID] = 1
                break
        if self.hasIntialized[documentID] == 0:
            return -1

        kChoosed = 0
        if self.hasIntialized[documentID] == 2:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        elif self.hasIntialized[documentID] == 1:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        return kChoosed

    # At first iteration, we must give all documents a cluster.
    def sampleCluster_first_improve(self, d, document, documentID, isInitial):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            # if self.m_z[cluster] == 1:
            #     prob[cluster] = 0
            #     continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        allProb = 0  # record the amount of all probabilities
        prob_normalized = []  # record normalized probabilities
        for k in range(self.K + 1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k] / allProb)
        if self.hasIntialized[documentID] == 0:
            self.hasIntialized[documentID] = 1
        for k in range(self.K + 1):
            if prob_normalized[k] > threshold_fix:
                self.hasIntialized[documentID] = 2
                break

        kChoosed = 0
        if self.hasIntialized[documentID] == 2:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        elif self.hasIntialized[documentID] == 1:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        
        # output the first and second largest probabilities that document belongs to the clusters
        # if isInitial:
        #     prob_normalized.sort()
        #     with open("probfile.txt", 'a') as outf:
        #         json_obj = {}
        #         json_obj['documentID'] = documentID
        #         json_obj['firstPro'] = prob_normalized[-1]
        #         if prob_normalized.__len__() > 1:
        #             json_obj['secondPro'] = prob_normalized[-2]
        #         else:
        #             json_obj['secondPro'] = 0.0
        #         json_obj['preClu_first'] = kChoosed
        #         json_obj['preClu_Pro'] = prob[kChoosed] / allProb
        #         outf.write(json.dumps(json_obj) + '\n')
        
        return kChoosed

    # At last iteration, we must fix all documents.
    def sampleCluster_iter_improve(self, d, document, documentID, isLast):
        prob = [float(0.0)] * (self.K + 1)
        for cluster in range(self.K):
            if cluster not in self.m_z or self.m_z[cluster] == 0:
                self.m_z[cluster] = 0
                prob[cluster] = 0
                continue
            # if self.m_z[cluster] == 1:
            #     prob[cluster] = 0
            #     continue
            prob[cluster] = self.m_z[cluster] / (self.D - 1 + self.alpha)
            valueOfRule2 = 1.0
            i = 0
            for w in range(document.wordNum):
                wordNo = document.wordIdArray[w]
                wordFre = document.wordFreArray[w]
                for j in range(wordFre):
                    if cluster not in self.n_zv:
                        self.n_zv = {}
                    if wordNo not in self.n_zv[cluster]:
                        self.n_zv[cluster][wordNo] = 0
                    if cluster not in self.n_z:
                        self.n_z[cluster] = 0
                    valueOfRule2 *= (self.n_zv[cluster][wordNo] + self.beta + j) / (self.n_z[cluster] + self.beta0 + i)
                    i += 1
            prob[cluster] *= valueOfRule2
        prob[self.K] = self.alpha / (self.D - 1 + self.alpha)
        valueOfRule2 = 1.0
        i = 0
        for w in range(document.wordNum):
            wordFre = document.wordFreArray[w]
            for j in range(wordFre):
                valueOfRule2 *= (self.beta + j) / (self.beta0 + i)
                i += 1
        prob[self.K] *= valueOfRule2

        allProb = 0 # record the amount of all probabilities
        prob_normalized = [] # record normalized probabilities
        for k in range(self.K+1):
            allProb += prob[k]
        for k in range(self.K + 1):
            prob_normalized.append(prob[k]/allProb)
        if self.hasIntialized[documentID] == 0:
            self.hasIntialized[documentID] = 1
        for k in range(self.K + 1):
            if prob_normalized[k] > threshold_fix:
                self.hasIntialized[documentID] = 2
                break

        if isLast:
            kChoosed = 0
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
            return kChoosed

        kChoosed = 0
        if self.hasIntialized[documentID] == 2:
            bigPro = prob[0]
            for k in range(1, self.K + 1):
                if prob[k] > bigPro:
                    bigPro = prob[k]
                    kChoosed = k
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        elif self.hasIntialized[documentID] == 1:
            for k in range(1, self.K + 1):
                prob[k] += prob[k - 1]
            thred = random.random() * prob[self.K]
            while kChoosed < self.K + 1:
                if thred < prob[kChoosed]:
                    break
                kChoosed += 1
            if kChoosed == self.K:
                self.K += 1
                self.K_current += 1
        return kChoosed

    def checkEmpty(self, cluster):
        if self.m_z[cluster] == 0:
            self.K_current -= 1

    def output_withAllBatchNum(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "BatchNum" + str(self.AllBatchNum) + "Batch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def output(self, documentSet, outputPath, wordList, batchNum):
        outputDir = outputPath + self.dataset + self.ParametersStr + "Batch" + str(batchNum) + "/"
        try:
            isExists = os.path.exists(outputDir)
            if not isExists:
                os.mkdir(outputDir)
                print("\tCreate directory:", outputDir)
        except:
            print("ERROR: Failed to create directory:", outputDir)
        self.outputClusteringResult(outputDir, documentSet)
        self.estimatePosterior()
        try:
            self.outputPhiWordsInTopics(outputDir, wordList, self.wordsInTopicNum)
        except:
            print("\tOutput Phi Words Wrong!")
        self.outputSizeOfEachCluster(outputDir, documentSet)

    def estimatePosterior(self):
        self.phi_zv = {}
        for cluster in self.n_zv:
            n_z_sum = 0
            if self.m_z[cluster] != 0:
                if cluster not in self.phi_zv:
                    self.phi_zv[cluster] = {}
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        n_z_sum += self.n_zv[cluster][v]
                for v in self.n_zv[cluster]:
                    if self.n_zv[cluster][v] != 0:
                        self.phi_zv[cluster][v] = float(self.n_zv[cluster][v] + self.beta) / float(n_z_sum + self.beta0)

    def getTop(self, array, rankList, Cnt):
        index = 0
        m = 0
        while m < Cnt and m < len(array):
            max = 0
            for no in array:
                if (array[no] > max and no not in rankList):
                    index = no
                    max = array[no]
            rankList.append(index)
            m += 1

    def outputPhiWordsInTopics(self, outputDir, wordList, Cnt):
        outputfiledir = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "PhiWordsInTopics.txt"
        writer = open(outputfiledir, 'w')
        for k in range(self.K):
            rankList = []
            if k not in self.phi_zv:
                continue
            topicline = "Topic " + str(k) + ":\n"
            writer.write(topicline)
            self.getTop(self.phi_zv[k], rankList, Cnt)
            for i in range(rankList.__len__()):
                tmp = "\t" + wordList[rankList[i]] + "\t" + str(self.phi_zv[k][rankList[i]])
                writer.write(tmp + "\n")
        writer.close()

    def outputSizeOfEachCluster(self, outputDir, documentSet):
        outputfile = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "SizeOfEachCluster.txt"
        writer = open(outputfile, 'w')
        topicCountIntList = []
        for cluster in range(self.K):
            if self.m_z[cluster] != 0:
                topicCountIntList.append([cluster, self.m_z[cluster]])
        line = ""
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n\n")
        line = ""
        topicCountIntList.sort(key = lambda tc: tc[1], reverse = True)
        for i in range(topicCountIntList.__len__()):
            line += str(topicCountIntList[i][0]) + ":" + str(topicCountIntList[i][1]) + ",\t"
        writer.write(line + "\n")
        writer.close()

    def outputClusteringResult(self, outputDir, documentSet):
        outputPath = outputDir + str(self.dataset) + "SampleNo" + str(self.sampleNo) + "ClusteringResult" + ".txt"
        writer = open(outputPath, 'w')
        for d in range(self.startDoc, self.currentDoc):
            documentID = documentSet.documents[d].documentID
            cluster = self.z[documentID]
            writer.write(str(documentID) + " " + str(cluster) + "\n")
        writer.close()


