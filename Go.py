__author__ = 'Lisa'
import csv
import os
import zipfile
import numpy
from sklearn import model_selection
from sklearn import feature_selection as fs
from sklearn import feature_extraction as fe
from scipy import stats
from matplotlib.mlab import PCA as mlabPCA
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import tree
from sklearn.svm import libsvm
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
class Go:
    #number of files in X
    fileNum = 0
    #Y data
    #response_data = 0
    #real_art_data = 0

    #rawFeatureList = 0
    featureList = 0
    lowClass = -1
    highClass = -1
    #response_rowNum = 0
    #real_art_rowNum = 0
    #accuracyOverall = 0
    #accuracyFirstClass = 0
    #accuracySecondClass = 0
    #bestFeatures = 0
    #accIncr = 0
    #subjMisclassified = 0
    #idx = 0
    #PCNumTOTAL = 0
    files = 0

    dataFiles = "/data"
    outputFolder = "/output"
    currentPath = ""

    def runElasticNet(self, X, y):
        alpha = [str(1e-20), str(1e-19), str(1e-18), str(1e-17), str(1e-16), str(1e-15), str(1e-14), str(1e-13),
                 str(1e-12), str(1e-11), str(1e-10), str(1e-9), str(1e-8), str(1e-7), str(1e-6), str(1e-5), str(1e-4),
                 str(1e-3), str(1e-2), str(1e-1), str(1e0), str(1e1), str(1e2), str(1e3), str(1e4), str(1e5)]
        l_ratio = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
        coef = numpy.zeros((alpha.len(), l_ratio.len()))
        for i in range(0, alpha.len()):
            for j in range(0, l_ratio.len()):
                en = SGDClassifier(loss="log", penalty="elasticnet", alpha=alpha[i], l1_ratio=l_ratio[j])
                en.fit(X, y)
                coef[i][j] = en.coef_

        #logical programming
        for i in range(0, alpha.len()):
            for j in range(0, l_ratio.len()):
                for k in range(0, coef[i][j].len()):
                    if coef[i][j][k] == -1:
                        coef[i][j][k] = 0

    def Lasso_new(self, X, y):
        # parameter tuning
        alphaarr = [0.07,0.009,0.008,0.001,0.09,0.08,0.985,0.085,0.05,0.01,0.1]
        print("++Sumit+ code-- LAsso")
        alphaarr.sort()
        print('alphaarr',alphaarr)

        ##LASSO CODE

        '''scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)'''

        # , alpha=float(alpha)
        best_result = []
        max_result = []
        min_result = []
        kfold = model_selection.KFold(n_splits=10)
        for alpha in alphaarr:
            clf = SGDClassifier(loss="log", penalty="l1", alpha=float(alpha))
            ress = clf.fit(X, y)
            model = fs.SelectFromModel(ress, prefit=True)
            X_new = model.transform(X)
            if X_new.shape[1]>2:
                cv_results = model_selection.cross_val_score(SGDClassifier(loss="log", penalty="l1", alpha=float(alpha)),
                                                         X_new, y,
                                                         cv=kfold, scoring='accuracy')
            ## check if train data can be scaled
            # print("cv_results",cv_results.mean(),'alpha --',alpha)
                best_result.append(cv_results.mean())
                max_result.append(cv_results.max())
                min_result.append(cv_results.min())

        bob = best_result.index(max(best_result))

        print('bob--', bob, 'best_result-', best_result[bob], 'alpha-', alphaarr[bob])
        print('max--',max_result[bob])
        print('max--',min_result[bob])

        clf = SGDClassifier(loss="log", penalty="l1", alpha=float(alphaarr[bob]))
        ress = clf.fit(X, y)
        model = fs.SelectFromModel(ress, prefit=True)
        X_new = model.transform(X)
        print(X_new.shape)
        regr=Lasso(alpha=alphaarr[bob])
        regr.fit(X,y)

        y_pred=regr.predict(X)
        print("Lasso score on training set: ", numpy.sqrt(mean_squared_error(y,y_pred)))
        #ret.extend([regr.coef_])
        #coef_list=list(regr.sparse_coef_.data)
        #coefic=pd.DataFrame(regr.sparse_coef_.toarray())
        print('sparse coef -',regr.sparse_coef_)
        coref_arr=numpy.array(regr.sparse_coef_.toarray())
        print(len(coref_arr[0]))
        proce=[]
        a={x:coref_arr[0][x] for x in range(len(coref_arr[0])) if coref_arr[0][x]!=0.0}

        #print(a[:3])

        #print('intercept  -',regr.intercept_)
        #print(type(coefic))
        #print(coefic(1,1))
        sorta=list(sorted(a, key=a.__getitem__, reverse=True))
        print(sorta)
        print(sorta[1])
        print(a.values()[1])



    def runLASSO(self, X, y):
        #parameter tuning
        alphaarr = [str(1e-20), str(1e-19), str(1e-18), str(1e-17), str(1e-16), str(1e-15), str(1e-14), str(1e-13),
                 str(1e-12), str(1e-11), str(1e-10), str(1e-9), str(1e-8), str(1e-7), str(1e-6), str(1e-5), str(1e-4),
                 str(1e-3), str(1e-2), str(1e-1), str(1e0), str(1e1), str(1e2), str(1e3), str(1e4), str(1e5)]

        #coef = numpy.zeros((alpha.len()))
        #for i in range(0, alpha.len()):
        #    clf = SGDClassifier(loss="log", penalty="l1", alpha=alpha[i])
        #    clf.fit(X, y)
        #    coef[i] = clf.coef_

        #logical programming
        #for i in range(0, alpha.len()):
        #    for k in range(0, coef[i].len()):
        #        if coef[i][k] == -1:
        #           coef[i][k] = 0


        ##X=X.tolist()
        ##X = [[int(float(j)) for j in i] for i in X]
        ##y = [int(float(i)) for i in y]
        print("++Sumit+ code-- LAsso")


        ##LASSO CODE
        regr=Lasso(alpha=0.15)
        regr.fit(X,y)
        y_pred=regr.predict(X)
        print("Lasso score on training set: ", numpy.sqrt(mean_squared_error(y,y_pred)))
        rss=sum((y_pred-y)**2)
        ret=[rss]
        ret.extend([regr.intercept_])
        #ret.extend([regr.coef_])
        print('sparse coef -',regr.sparse_coef_)
        print('intercept  -',regr.intercept_)

        #foldNo = 10
        ## chose which fold have best accuracy
        results=[]
        #skf = StratifiedKFold(y, n_folds=foldNo)
        #print("skf--",skf)
        #print("X =", X, "\nY = ", y)

        '''for train_index, test_index in skf:
            X_train, X_test = numpy.array(X[train_index]), numpy.array(X[test_index])
            print("XTrain=", X_train, "\nXTest=", X_test)

            y=numpy.array(y)
            y_train = numpy.array(y[train_index])
            y_test = numpy.array(y[test_index])
            print("\ny_train=", y_train, "\ny_test=", y_test)

            clf = SGDClassifier(loss="log", penalty="l1", alpha=0.1)
            clf.fit(X_train, y_train)
            print('clf in loop is', clf)
            y_predSGD=clf.predict(X)
            results.append(y_predSGD)
        '''
        scaler=StandardScaler()
        scaler.fit(X)
        X=scaler.transform(X)
        #, alpha=float(alpha)
        best_result=[]
        kfold = model_selection.KFold(n_splits=10)
        for alpha in alphaarr:
            cv_results = model_selection.cross_val_score(SGDClassifier(loss="log", penalty="l1", alpha=float(alpha)), X, y, cv=kfold, scoring='accuracy')
            ## check if train data can be scaled
            #print("cv_results",cv_results.mean(),'alpha --',alpha)
            best_result.append(cv_results.mean())
        bob=best_result.index(max(best_result))
        print('bob--',bob,'best_result-',best_result[bob],'alpha-',alphaarr[bob])
        ##y_predicted = StratifiedKFold(X, n_folds=foldNo, shuffle=False, random_state=None)
        #accuracyCalculation(y_predicted, Go.lowclass, instOrder)

        #print("results---",results)
        #use parameters with highest overall accuracy

        #clf = SGDClassifier(loss="log", penalty="l1", alpha=float(alphaarr[bob]))
        clf = SGDClassifier(loss="log", penalty="l1", alpha=0.08)
        ress=clf.fit(X, y)
        #print('clf predict',clf.predict(X))
        #print('score---',clf.score(X,y))
        #print('sparcify---',clf.sparsify())
        #model = fs.SelectFromModel(SGDClassifier(loss="log", penalty="l1", alpha=float(alphaarr[bob])).fit(X, y),prefit=True)
        model=fs.SelectFromModel(ress,prefit=True)
        X_new=model.transform(X)
        print('orgin --',X.shape)
        print('new --',X_new.shape)
        ret=clf.predict(X)

        indx_ret1=[index for index, value in enumerate(ret) if value == 1]
        indx_y1=[index for index, value in enumerate(y) if value == 1]

        indx_ret0=[index for index, value in enumerate(ret) if value == 0]
        indx_y0=[index for index, value in enumerate(y) if value == 0]

        print('len total',len(list(set(indx_ret0)&set(indx_y0)))+len(list(set(indx_ret1)&set(indx_y1))))


        return {"bestFeatureSummary": indx_ret1}

    def processFiles(self, csvResponse, csvRealArt, datazipfilepath):
        print('==sumit==',datazipfilepath)
        print('==csvResponse==', csvResponse)
        print('==csvRealArt==', csvRealArt)
        #create output files in output folder
        response = csv.reader(open(csvResponse))
        real_art = csv.reader(open(csvRealArt))

        response_data = list(response)
        real_art_data = list(real_art)

        for i in range(len(response_data)):
            response_data[i] = float(response_data[i][0])

        for i in range(len(real_art_data)):
            real_art_data[i] = float(real_art_data[i][0])

        response_rowNum = len(response_data)
        real_art_rowNum = len(real_art_data)

        Go.lowClass = -1
        Go.highClass = -1

        for row in range(0, response_rowNum):
            if row == 0:
                Go.lowClass = response_data[row]
            elif response_data[row] != Go.lowClass:
                Go.highClass = response_data[row]
                if Go.highClass < Go.lowClass:
                    Go.lowClass = Go.highClass
                break

        #get current path
        Go.currentPath = os.getcwd()
        datazip = zipfile.ZipFile(datazipfilepath, 'r')
        datazip.extractall(Go.currentPath + Go.dataFiles)
        os.chdir(Go.currentPath + Go.dataFiles + "/datazip")
        Go.files = os.listdir()
        for i in range(0, len(Go.files)):
            print(Go.files[i])
            if Go.files[i].endswith('.csv') == False:
                Go.files.pop(i)
        print(Go.files)
        sortedFiles = sorted(Go.files)
        Go.fileNum = len(sortedFiles)

        #create empty list rawFeatureList
        rawFeatureList = numpy.empty(Go.fileNum, dtype=numpy.ndarray)
        #rawFeatureList = numpy.zeros([fileNum, 10000])

        listt=[]
        matrix=numpy.ndarray(shape=(106,0),dtype=float,order='F')

        for i in range(0, Go.fileNum):
            data = numpy.genfromtxt(Go.files[i], dtype=float, delimiter=",")
            rawFeatureList[i] = data
            matrix=numpy.concatenate((matrix,data),axis=1)

        #rawFeatureList to features var ready for USFS
        #scoreTotal = numpy.zeros((response_rowNum, PCNumTOTAL))
        features = []
        for i in range(0, Go.fileNum):
            if features == []:
                features = rawFeatureList[i]
            else:
                features = numpy.hstack((features, rawFeatureList[i]))
                #features.append(rawFeatureList[i])

        return {'response': response_data, 'realArt': real_art_data, 'dataList': rawFeatureList, 'data': matrix}

    def DimReduction(self, varToKeep, response_data, rawFeatureList):
        response_rowNum = len(response_data) #length of response
        Go.featureList = numpy.empty(Go.fileNum, dtype=numpy.ndarray)
        scoreList = [0]*Go.fileNum
        PCNoList = numpy.empty(Go.fileNum, dtype=int)
        coeffList = [0]*Go.fileNum
        os.chdir(Go.currentPath + Go.dataFiles + Go.outputFolder)
        Go.featureListArray = numpy.empty(Go.fileNum, dtype=numpy.ndarray)
        for i in range(Go.fileNum):
            Go.featureList[i] = stats.zscore(rawFeatureList[i])
            #if i == 0:
            #    Go.featureListArray = Go.featureList[i]
            #else:
            #    Go.featureListArray = numpy.vstack((Go.featureListArray, Go.featureList[i]))

        VarianceIncluded = "Variance Included is: "
        for featNum in range(Go.fileNum):
            #print ("===Sumit:===",featNum,"++",Go.featureList[featNum])
            PCAobject = mlabPCA(Go.featureList[featNum], standardize=False)
            explained = 100 * PCAobject.fracs # this is correct
            coeff = PCAobject.Wt.T #this is correct, except last column has +/- signs switched
            score = PCAobject.Y #same issue as coeff (but i dont think its significant?)

            i = 0
            j = 0
            k = 0

            while i < len(explained):
                j = j + explained[i]
                k = i
                if j > varToKeep:
                    break
                i += 1

            scoreList[featNum] = score[:, 0:k+1]
            coeffList[featNum] = coeff[:, 0:k+1]
            PCNoList[featNum] = k+1
            '''
            print("Coeff is ")
            print(coeff)
            print("Score is:")
            print(score)
            print("Explained is:")
            print(explained)
            '''
            string1 = 'CoeffMatrix' + Go.files[featNum]
            string2 = 'ScoreMatrix' + Go.files[featNum]

            file1 = open(string1, 'wb')
            wr1 = csv.writer(file1, quoting=csv.QUOTE_ALL)

            numpy.savetxt(string1, coeffList[featNum], delimiter=",")

            file2 = open(string2, 'wb')
            wr2 = csv.writer(file2, quoting=csv.QUOTE_ALL)
            numpy.savetxt(string2, scoreList[featNum], delimiter=",")

        PCNumTOTAL = sum(PCNoList)
        PCNumCum = numpy.cumsum(PCNoList)

        file_PCNumCum = open('PCNumCum.csv', 'wb')
        wr3 = csv.writer(file_PCNumCum, quoting=csv.QUOTE_ALL)

        numpy.savetxt('PCNumCum.csv', PCNumCum, delimiter=",")

        scoreTotal = numpy.zeros((response_rowNum, PCNumTOTAL))

        x = 0

        for i in range(0, Go.fileNum):
            numRowsScoreList = len(scoreList[i])
            numColScoreList = len(scoreList[i][0])
            print(numColScoreList)
            scoreTotal[:, x:x+numColScoreList] = scoreList[i]
            x += numColScoreList

        file_PCScoreTotal = open('PCScoreTotal.csv', 'wb')
        wr4 = csv.writer(file_PCScoreTotal, quoting=csv.QUOTE_ALL)
        numpy.savetxt('PCScoreTotal.csv', scoreTotal, delimiter=",")

        if featNum == 0:
            VarianceIncluded += str(j)
        else:
            VarianceIncluded += ", " + str(j)

        return {'VarianceIncluded': VarianceIncluded, 'scoreTotal': scoreTotal}

    def classifierTrainTest(score, diagn, real_art, cvPartition, classifier, subjIndex, preAccMatrix, preInstOrder):
        x = 0
        iteration = 0
        idx = 0
        PCNo = len(score[0])
        subAccMatrix = 0
        # FIX: what is test->matlab function within cvpartition class
        #idx = numpy.random.rand(cvPartition, iteration)
        #idx_test = numpy.where(idx == 1)
        #idx_train = numpy.where(idx != 1)

        #QUESTION: cv partition not scalar ,how works
        #iteration must be atleast 2
        for idx_train, idx_test in cvPartition:
            #change idx to boolean array
            idx = numpy.zeros((len(score), 1), dtype=bool)
            for index in idx_test:
                idx[index] = True

            #for testing purposes
            #idx = numpy.zeros((len(score), 1), dtype=bool)
            #idx[47] = True

            #idx is all training in MATLAB implementation?
            cvTEST = numpy.zeros((sum(idx), PCNo))
            diagnTEST = numpy.zeros((sum(idx), 1))
            real_artTEST = numpy.zeros((sum(idx), 1))
            instIndexTEST = numpy.zeros((sum(idx), 1))

            cvTRAIN = numpy.zeros((len(idx) - sum(idx), PCNo))
            diagnTRAIN = numpy.zeros((len(idx) - sum(idx), 1))
            real_artTRAIN = numpy.zeros((len(idx) - sum(idx), 1))

            k = 0
            m = 0

            for j in range(len(idx)):
                if idx[j] == 1:
                    cvTEST[k, :] = score[j, :]
                    diagnTEST[k] = diagn[j]
                    real_artTEST[k] = real_art[j]
                    instIndexTEST[k] = subjIndex[j]
                    k = k + 1
                else:
                    cvTRAIN[m, :] = score[j, :]
                    diagnTRAIN[m] = diagn[j]
                    real_artTRAIN[m] = real_art[j]
                    m = m + 1


            # FIX: use scikit-learn for classifiers and predictions
            if classifier == "lda":
                #ldaModel = LDA()
                priorsArrays = numpy.array((.5, .5))
                ldaModel = LDA(solver='eigen', priors=priorsArrays, shrinkage=1.00)
                #ldaModel = LDA()
                ldaModel.fit(cvTRAIN, diagnTRAIN)
                label = ldaModel.predict(cvTEST)
            elif classifier == 'qda':
                # training a quadratic discriminant classifier to the data
                qdaModel = QDA()
                priorsArrays = numpy.array((.5, .5))
                #qdaModel = QDA(solver='eigen', priors=priorsArrays, shrinkage=1.00)
                qdaModel.fit(cvTRAIN, diagnTRAIN)
                label = qdaModel.predict(cvTEST)
            elif classifier == 'tree':
                # training a decision tree to the data
                treeModel = tree()
                treeModel.fit(cvTRAIN, diagnTRAIN)
                label = treeModel.predict(cvTEST)
            elif classifier == 'svm':
                # training a support vector machine to the data
                svmModel = SVC()
                svmModel.fit(cvTRAIN, diagnTRAIN)
                label = svmModel.predict(cvTEST)

            trueClassLabel = diagnTEST
            predictedClassLabel = label

            #from former loop

            subAccMatrix = numpy.column_stack((trueClassLabel, predictedClassLabel, real_artTEST))
            preAccMatrix[x:x + len(subAccMatrix[:, 0]), :] = subAccMatrix
            preInstOrder[x:x + len(instIndexTEST[:, 0])] = instIndexTEST

            x = x + len(subAccMatrix[:, 0])

            #for testing purposes
            #break
        # create dictionary for return values
        return {'cvTEST': cvTEST, 'diagnTEST': diagnTEST, 'real_artTEST': real_artTEST, 'instIndexTEST': instIndexTEST,
                'cvTRAIN': cvTRAIN, 'diagnTRAIN': diagnTRAIN, 'real_artTRAIN': real_artTRAIN,
                'trueClassLabel': trueClassLabel, 'predictedClassLabel': predictedClassLabel, 'idx': idx,
                'subAccMatrix': subAccMatrix, 'preAccMatrix': preAccMatrix, 'preInstOrder': preInstOrder}

    def accuracyCalculation(accMatrix, lowClass, instOrder):
        element1 = 0
        element2 = 0
        misclassified1 = 0
        misclassified2 = 0

        instMisclass = numpy.column_stack((instOrder, numpy.zeros((len(instOrder), 1))))

        for i in range(len(accMatrix[:, 0])):
            if accMatrix[i][0] == lowClass:
                element1 = element1 + 1
            else:
                element2 = element2 + 1

            if accMatrix[i][0] != accMatrix[i][1]:
                if accMatrix[i][0] == lowClass:
                    misclassified1 = misclassified1 + 1
                else:
                    misclassified2 = misclassified2 + 1
                instMisclass[i][1] = 1

        accuracy = (1 - (misclassified1 + misclassified2) / (element1 + element2)) * 100
        lowClassAccuracy = (1 - misclassified1 / element1) * 100
        highClassAccuracy = (1 - misclassified2 / element2) * 100

        return {'accuracy': accuracy, 'lowClassAccuracy': lowClassAccuracy, 'highClassAccuracy': highClassAccuracy,
                'instMisclass': instMisclass}

    def runUSFS(self, response_data, real_art_data, features, classifierType):
        print("USFS function called\n")

        response_rowNum = len(response_data)
        real_art_rowNum = len(real_art_data)
        realStatus = 1
        cvStatus = 1

        classifNo = len(classifierType)

        if cvStatus == 0:
            foldNo = 10
            iterationLength = 10
        else:
            foldNo = response_rowNum
            iterationLength = 1

        instanceIndex = numpy.zeros((response_rowNum, 1))
        for i in range(0, instanceIndex.size):
            instanceIndex[i] = i

        if realStatus == 1:
            realInstanceIndex = numpy.zeros((real_art_rowNum, 1))
            for i in range(realInstanceIndex.size):
                realInstanceIndex[i] = i

        accuracyOverall = numpy.zeros((classifNo, 1))
        accuracyFirstClass = numpy.zeros((classifNo, 1))
        accuracySecondClass = numpy.zeros((classifNo, 1))
        bestFeatures = [0] * classifNo
        accIncr = [0] * classifNo
        subjMisclassified = numpy.array([classifNo, 1, iterationLength], dtype=object)
        idx = 0

        if cvStatus == 0:
            idx = numpy.zeros(classifNo, 1)

        for i in range(classifNo):
            accuracyOverall[i] = numpy.zeros((1, iterationLength))
            accuracyFirstClass[i] = numpy.zeros((1, iterationLength))
            accuracySecondClass[i] = numpy.zeros((1, iterationLength))

            bestFeatures[i] = [0] * iterationLength
            accIncr[i] = [0] * iterationLength

            if realStatus == 0:
                subjMisclassified[i] = numpy.concatenate((instanceIndex,
                                                          numpy.zeros((response_rowNum, iterationLength))), axis=1)
            else:
                subjMisclassified[i] = numpy.concatenate((realInstanceIndex,
                                                          numpy.zeros((real_art_rowNum, iterationLength))), axis=1)
            if cvStatus == 0:
                idx[i] = numpy.zeros((1, iterationLength))
                for j in range(iterationLength):
                    idx[i][j] = numpy.zeros((foldNo, 1))

        bestFeatureIndex = numpy.array([])
        accIncrTracker = numpy.array([])

        for z1 in range(iterationLength):
            print('Iteration ' + str(z1))
            #PCA Stuff was here

            cvPartition = -1
            if cvStatus == 0:
                #cvPartition = StratifiedKFold(Go.response, n_folds=foldNo, shuffle=False, random_state=None)
                cvPartition = StratifiedKFold(response_data, n_folds=foldNo, shuffle=False, random_state=None)

            else:
                cvPartition = LeaveOneOut(len(response_data))

            #print("PCNumTOTAL:")
            #print(Go.PCNumTOTAL)
            
            #featureNumTOTAL is equivaluent to PCNumTOTAL
            featShape = features.shape
            featureNumTOTAL = featShape[1]
            featureIndexNumbers = numpy.zeros((featureNumTOTAL, 1))
            for i in range(0, featureNumTOTAL):
                featureIndexNumbers[i] = i

            for z2 in range(0, classifNo):
                print('Classifier ' + classifierType[z2])
                classifier = classifierType[z2]

                maxAcc = 0
                #scorebestFeatures = numpy.zeros(scoreTotal.shape)
                scorebestFeatures = []
                bestFeatureIndex = numpy.array([])


                featureNoTOTAL = featureNumTOTAL
                scoreFeatureTOTAL = features
                featureIndexNo = featureIndexNumbers

                maxAccTracker = numpy.array([0, 100])
                maxAccIndex = 0
                maxAccuracy = 0
                lowClassAccuracy = 0
                highClassAccuracy = 0

                finalInstMisclass = []

                lowClassAccuracies = []
                highClassAccuracies = []
                instMisclass = []

                lt = 1

                while (maxAccTracker[1] - maxAccTracker[0]) > 1:
                    if lt > 1:

                        if scorebestFeatures != []:
                            scorebestFeatures = numpy.column_stack((scorebestFeatures, scoreFeatureTOTAL[:, maxAccIndex]))
                        else:
                            scorebestFeatures = scoreFeatureTOTAL[:, maxAccIndex]
                        if bestFeatureIndex.size != 0:
                            bestFeatureIndex = numpy.append(bestFeatureIndex, [featureIndexNo[maxAccIndex]])
                            #bestFeatureIndex = numpy.append((bestFeatureIndex, featureIndexNo[maxAccIndex]))
                        else:
                            bestFeatureIndex = featureIndexNo[maxAccIndex]

                        #should be 0?
                        if maxAccIndex == 1:
                            scoreFeatureTOTAL = scoreFeatureTOTAL[:, 1:featureNoTOTAL]
                            featureIndexNo = featureIndexNo[1:featureNoTOTAL]

                        elif maxAccIndex == featureNoTOTAL-1:
                            scoreFeatureTOTAL = scoreFeatureTOTAL[:, 0:featureNoTOTAL - 1]
                            featureIndexNo = featureIndexNo[0:featureNoTOTAL - 1]
                        else:
                            scoreFeatureTOTAL = numpy.column_stack((scoreFeatureTOTAL[:, 0:maxAccIndex],
                                                            scoreFeatureTOTAL[:, maxAccIndex+1:featureNoTOTAL]))
                            featureIndexNo = numpy.row_stack((featureIndexNo[0:maxAccIndex],
                                                           featureIndexNo[maxAccIndex+1:featureNoTOTAL]))

                        lowClassAccuracy = lowClassAccuracies[0][maxAccIndex]
                        highClassAccuracy = highClassAccuracies[0][maxAccIndex]

                        finalInstMisclass = instMisclass[maxAccIndex]


                        #numpy function for row concatenation
                        if accIncrTracker.size != 0:
                            accIncrTracker = numpy.append(accIncrTracker, [(maxAccTracker[1] - maxAccTracker[0])])
                        else:
                            accIncrTracker = numpy.array([maxAccTracker[1]-maxAccTracker[0]])

                        maxAccuracy = maxAcc

                        featureNoTOTAL = featureNoTOTAL - 1
                        #end not checked

                    accuracies = numpy.zeros((1, featureNoTOTAL))
                    lowClassAccuracies = numpy.zeros((1, featureNoTOTAL))
                    highClassAccuracies = numpy.zeros((1, featureNoTOTAL))
                    #instMisclass = numpy.zeros((1, featureNoTOTAL))
                    instMisclass = [0] * featureNoTOTAL

                    for i in range(0, featureNoTOTAL):
                        #numpy function for column concatenation
                        #print(scorebestFeatures.shape)
                        #print(scoreTotal.shape)
                        scoreCandidateFeatures = 0
                        if scorebestFeatures != []:
                            scoreCandidateFeatures = numpy.column_stack((scorebestFeatures, scoreFeatureTOTAL[:, i]))
                        else:
                            scoreCandidateFeatures = numpy.reshape(features[:, i], (len(features), 1))

                        preAccMatrix = numpy.zeros((len(scoreCandidateFeatures), 3))
                        preInstOrder = numpy.zeros((len(scoreCandidateFeatures), 1))

                        #x = 0     put in classifierTrainTest
                            #FIX: lines 280-285

                        #for j in range(0, foldNo):        put loop in classifierTrainTest
                        if cvStatus == 0:
                            Go.classifierTrainTest(scoreCandidateFeatures, response_data, real_art_data, cvPartition,
                                                   classifier, instanceIndex, preAccMatrix, preInstOrder)
                            real_artTEST = dict.get('real_artTEST')
                            instIndexTEST = dict.get('instIndexTEST')
                            trueClassLabel = dict.get('trueClassLabel')
                            predictedClassLabel = dict.get('predictedClassLabel')
                            #return all of idx[j] to idx[z2][z1]
                            idx[z2][z1][j] = dict.get('idx')

                        else:
                            dict = Go.classifierTrainTest(scoreCandidateFeatures, response_data, real_art_data,
                                                          cvPartition, classifier, instanceIndex, preAccMatrix,
                                                          preInstOrder)
                            real_artTEST = dict.get('real_artTEST')
                            instIndexTEST = dict.get('instIndexTEST')
                            trueClassLabel = dict.get('trueClassLabel')
                            predictedClassLabel = dict.get('predictedClassLabel')

                            subAccMatrix = dict.get('subAccMatrix')
                            preAccMatrix = dict.get('preAccMatrix')
                            preInstOrder = dict.get('preInstOrder')

                        if realStatus == 1:
                            accMatrix = numpy.zeros((sum(preAccMatrix[:, 2]), 2))
                            instOrder = numpy.zeros((sum(preAccMatrix[:, 2]), 1))
                            j = 0
                            for k in range(len(preAccMatrix[:, 2])):
                                if preAccMatrix[k, 2] == 1:
                                    accMatrix[j, 0:2] = preAccMatrix[k, 0:2]
                                    instOrder[j] = preInstOrder[k]
                                    j = j + 1
                        else:
                            accMatrix = preAccMatrix[:, 0:2]
                            instOrder = preInstOrder

                        dict2 = Go.accuracyCalculation(accMatrix, Go.lowClass, instOrder)
                        accuracies[0][i] = dict2.get('accuracy')
                        lowClassAccuracies[0][i] = dict2\
                            .get('lowClassAccuracy')
                        highClassAccuracies[0][i] = dict2.get('highClassAccuracy')
                        instMisclass[i] = dict2.get('instMisclass')

                    maxAccIndex = numpy.argmax(accuracies)
                    maxAcc = numpy.amax(accuracies)

                    if (maxAccTracker[0] == 0) and (maxAccTracker[1] == 100):
                        maxAccTracker = numpy.array([0, maxAcc])
                    else:
                        maxAccTracker[0] = maxAccTracker[1]
                        maxAccTracker[1] = maxAcc

                    if (featureNoTOTAL == 1) and ((maxAccTracker[1] - maxAccTracker[0]) > 1):
                        scorebestFeatures = numpy.column_stack((scorebestFeatures, scoreFeatureTOTAL))
                        bestFeatureIndex = numpy.hstack((bestFeatureIndex, featureIndexNo))
                        scoreFeatureTOTAL = []
                        featureIndexNo = []

                        lowClassAccuracy = lowClassAccuracies
                        highClassAccuracy = highClassAccuracies

                        finalInstMisclass = instMisclass[maxAccIndex]

                        accIncrTracker = numpy.hstack((accIncrTracker, maxAccTracker[1] - maxAccTracker[0]))
                        maxAccuracy = maxAcc
                        maxAccTracker = numpy.matrix['0, 0.5']

                    lt += 1

                print("Out of loop")

                print(finalInstMisclass)
                #order = numpy.argsort(finalInstMisclass[:, 0])
                #for i in range(0, len(order)):
                #    finalInstMisclass = finalInstMisclass[order[i], :]
                finalInstMisclass = finalInstMisclass[:, 1]

                # FIX: curly brackets vs parentheses? lines 359-364
                accuracyOverall[z2][z1] = maxAccuracy
                accuracyFirstClass[z2][z1] = lowClassAccuracy
                accuracySecondClass[z2][z1] = highClassAccuracy
                bestFeatures[z2][z1] = bestFeatureIndex
                accIncr[z2][z1] = accIncrTracker
                # FIX: 3d array where you can replace a whole column
                shapeSubjMisclassified = subjMisclassified[z2].shape
                for c in range(0, shapeSubjMisclassified[0]):
                    subjMisclassified[z2][c][1 + z1] = finalInstMisclass[c]

        maxVal = numpy.zeros(classifNo)
        for i in range(0, classifNo):
            for j in range(0, iterationLength):
                bestFeaturesShape = bestFeatures[i][j].shape
                if bestFeaturesShape[0] > maxVal[i]:
                    maxVal[i] = bestFeaturesShape[0]

        bestFeaturesummary = [0]*classifNo
        # FIX: curly brackets vs parentheses? lines 387-391
        for i in range(0, classifNo):
            bestFeaturesummary[i] = numpy.zeros((3 + maxVal[i], iterationLength * 2))
            bestFeaturesummary[i][0, 0: iterationLength] = accuracyOverall[i]
            bestFeaturesummary[i][1, 0: iterationLength] = accuracyFirstClass[i]
            bestFeaturesummary[i][2, 0: iterationLength] = accuracySecondClass[i]

        for i in range(0, iterationLength):
            for j in range(0, classifNo):
                x = 3
                bestFeaturesShape = bestFeatures[j][i].shape
                for k in range(0, bestFeaturesShape[0]):
                    bestFeaturesummary[j][x][i] = bestFeatures[j][i][k]
                    bestFeaturesummary[j][x][i + iterationLength] = accIncr[j][i][k]
                    x += 1
        os.chdir(Go.currentPath + Go.dataFiles + Go.outputFolder)
        for i in range(0, classifNo):
            summary_string = 'SummarybestFeatures_' + classifierType[i] + '.csv'
            misclassified_string = 'MisclassifiedSubjects_' + classifierType[i] + '.csv'
            file3 = open(summary_string, 'wb')
            file4 = open(misclassified_string, 'wb')
            numpy.savetxt(file3, bestFeaturesummary[i], delimiter=',')
            numpy.savetxt(file4, subjMisclassified[i], delimiter=',')
            file3.close()
            file4.close()

        return {"bestFeatureSummary": bestFeaturesummary}




