import csv
import os
import numpy
import zipfile
import sklearn
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import LeaveOneOut
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.tree import tree
from sklearn.svm import libsvm
from sklearn.svm import SVC
from matplotlib.mlab import PCA as mlabPCA
import matplotlib.mlab
from scipy import stats
from scipy import linalg

class USFS:
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
        print("cvPartition:")
        print(cvPartition)

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
                #qdaModel = QDA()
                priorsArrays = numpy.array((.5, .5))
                qdaModel = QDA(solver='eigen', priors=priorsArrays, shrinkage=1.00)
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
        #end at line 487


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

    def printla(self):
        return "In USFS"
    def run(self, csvResponse, csvRealArt, datazipfilepath):
        print("USFS function called\n")
        #TEST
        #c = csv.writer(open('C:/Users/Lisa/PycharmProjects/HonorsThesis/MYFILE.csv', "wb"))
        #c.writerow(["Name","Address","Telephone","Fax","E-mail","Others"])


        # clear variables and associated memories
        #initialized variables
        #PCNoTOTAL = 0


        # create data files in data folder (string for name)
        dataFiles = "\data"
        outputFolder = "\output"

        #create output files in output folder
        response = csv.reader(open(csvResponse))
        real_art = csv.reader(open(csvRealArt))
        #response = csvResponse
        #real_art = csvRealArt

        response_data = list(response)
        real_art_data = list(real_art)

        for i in range(len(response_data)):
            response_data[i] = float(response_data[i][0])

        for i in range(len(real_art_data)):
            real_art_data[i] = float(real_art_data[i][0])

        #response_data = [float(i) for i in response_data]
        #real_art_data = [float(i) for i in real_art_data]


        response_rowNum = len(response_data)
        real_art_rowNum = len(real_art_data)

        lowClass = -1
        highClass = -1
        #FIX: read csv files "response.csv" and "real_art.csv"

        for row in range(1, response_rowNum):
            if row == 1:
                lowClass = response_data[row]
            elif response_data[row] != lowClass:
                highClass = response_data[row]
                if highClass < lowClass:
                    lowClass = highClass
                break

        #get current path
        currentPath = os.getcwd()
        datazip = zipfile.ZipFile(datazipfilepath, 'r')
        datazip.extractall(currentPath + dataFiles)
        os.chdir(currentPath + dataFiles + "\datazip")
        files = os.listdir()
        for i in range(0, len(files)):
            print(files[i])
            if files[i].endswith('.csv') == False:
                #print("HIT")
                files.pop(i)
        print(files)
        sortedFiles = sorted(files)
        fileNum = len(sortedFiles)

        #create empty list rawFeatureList
        rawFeatureList = numpy.empty(fileNum, dtype=numpy.ndarray)
        #rawFeatureList = numpy.zeros([fileNum, 10000])

        for i in range(0, fileNum):
            data = numpy.genfromtxt(files[i], dtype=float, delimiter=",")
            rawFeatureList[i] = data

        realStatus = 1
        cvStatus = 1

        classifierType = ["lda", "qda", "svm"]

        classifNo = len(classifierType)

        if cvStatus == 0:
            foldNo = 10
            iterationLength = 10
        else:
            foldNo = response_rowNum
            iterationLength = 1

        #QUESTION: why do you set each index to index?
        instanceIndex = numpy.zeros((response_rowNum, 1))
        for i in range(0, instanceIndex.size):
            instanceIndex[i] = i

        if realStatus == 1:
            realInstanceIndex = numpy.zeros((real_art_rowNum, 1))
            for i in range(realInstanceIndex.size):
                realInstanceIndex[i] = i

        # FIX: change accuracyOverall to 2d array
        accuracyOverall = numpy.zeros((classifNo, 1))
        accuracyFirstClass = numpy.zeros((classifNo, 1))
        accuracySecondClass = numpy.zeros((classifNo, 1))
        #bestPCS = numpy.zeros((classifNo, 1))
        bestPCS = [0] * classifNo
        accIncr = [0] * classifNo
        subjMisclassified = numpy.array([classifNo, 1, iterationLength], dtype=object)

        if cvStatus == 0:
            idx = numpy.zeros(classifNo, 1)

        # QUESTION: should these arrays be array of arrays?
        for i in range(classifNo):
            accuracyOverall[i] = numpy.zeros((1, iterationLength))
            accuracyFirstClass[i] = numpy.zeros((1, iterationLength))
            accuracySecondClass[i] = numpy.zeros((1, iterationLength))
            #bestPCS[i] = numpy.zeros((1, iterationLength))
            bestPCS[i] = [0] * iterationLength
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

        bestPCIndex = numpy.array([])
        accIncrTracker = numpy.array([])

        for z1 in range(iterationLength):
            print('Iteration ' + str(z1))
            #begin PCA
            featureList = numpy.empty(fileNum, dtype=numpy.ndarray)

            # import from scipy stats
            # FIX
            for i in range(fileNum):
                featureList[i] = stats.zscore(rawFeatureList[i])

            #scoreList = numpy.empty(fileNum, dtype=numpy.ndarray)
            scoreList = [0]*fileNum
            PCNoList = numpy.empty(fileNum, dtype=int)
            coeffList = [0]*fileNum

            os.chdir(currentPath + dataFiles + outputFolder)

            #import:  from matplotlib.mlab import PCA

            for featNum in range(0, fileNum):
                #PCAobject = PCA(featureList[featNum])
                #PCAobject = PCA(n_components=len(featureList[featNum][0, :]), copy=True, whiten=False)
                #X = numpy.matrix('1 30 2 4; 2 50 4 10; 8 20 2 3; 7 70 7 5; 2 10 3 9')
                #PCAobject.fit_transform(featureList[featNum])
                #print("X")
                #print(X)
                PCAobject = mlabPCA(featureList[featNum], standardize=False)
                i = 0
                j = 0

                explained = 100 * PCAobject.fracs # this is correct
                coeff = PCAobject.Wt.T #this is correct, except last column has +/- signs switched
                score = PCAobject.Y #same issue as coeff (but i dont think its significant?)
                print("Coeff is ")
                print(coeff)
                print("Score is:")
                print(score)
                print("Explained is:")
                print(explained)

                #print("featureList[featNum] is ")
                #print(featureList[featNum])

                print("PCA percentages: ")
                k = 0
                while i < len(explained):
                    j = j + explained[i]
                    k = i
                    if j > 85:
                        break
                    i += 1

                scoreList[featNum] = score[:, 0:k+1]
                coeffList[featNum] = coeff[:, 0:k+1]
                PCNoList[featNum] = k+1

                string1 = 'CoeffMatrix' + files[featNum]
                string2 = 'ScoreMatrix' + files[featNum]

                file1 = open(string1, 'wb')
                wr1 = csv.writer(file1, quoting=csv.QUOTE_ALL)
                #wr1.writerows(coeffList[featNum])
                numpy.savetxt(string1, coeffList[featNum], delimiter=",")

                file2 = open(string2, 'wb')
                wr2 = csv.writer(file2, quoting=csv.QUOTE_ALL)
                #wr2.writerows(scoreList[featNum])
                numpy.savetxt(string2, scoreList[featNum], delimiter=",")

            PCNumTOTAL = sum(PCNoList)
            PCNumCum = numpy.cumsum(PCNoList)

            file_PCNumCum = open('PCNumCum.csv', 'wb')
            wr3 = csv.writer(file_PCNumCum, quoting=csv.QUOTE_ALL)
            #wr3.writerows(PCNumCum)
            numpy.savetxt('PCNumCum.csv', PCNumCum, delimiter=",")

            scoreTotal = numpy.zeros((response_rowNum, PCNumTOTAL))
            x = 0

            for i in range(0, fileNum):
                #get shape of scoreList[i]
                numRowsScoreList = len(scoreList[i])
                numColScoreList = len(scoreList[i][0])
                print(numColScoreList)
                scoreTotal[:, x:x+numColScoreList] = scoreList[i]
                x += numColScoreList

            file_PCScoreTotal = open('PCScoreTotal.csv', 'wb')
            wr4 = csv.writer(file_PCScoreTotal, quoting=csv.QUOTE_ALL)
            numpy.savetxt('PCScoreTotal.csv', scoreTotal, delimiter=",")

            #end of PCA
            cvPartition = -1
            if cvStatus == 0:
                #to FIX
                #cvPartition = cvpartition(response, 'KFold', foldNo)
                cvPartition = StratifiedKFold(response, n_folds=foldNo, shuffle=False, random_state=None)
                #cvPartition = StratifiedKFold(response_data, n_folds=foldNo, shuffle=False, random_state=None)
                #numpy.random.shuffle(cvPartition)

            else:
                # to FIX
                #cvPartition = cvpartition(foldNo, 'LeaveOut')

                cvPartition = LeaveOneOut(len(response_data))
                #cvPartition = LeaveOneOut(foldNo)

                #numpy.random.shuffle(cvPartition)

            print("PCNumTOTAL:")
            print(PCNumTOTAL)
            pcIndexNumbers = numpy.zeros((PCNumTOTAL, 1))
            for i in range(0, PCNumTOTAL):
                pcIndexNumbers[i] = i

            for z2 in range(0, classifNo):
                print('Classifier ' + classifierType[z2])
                classifier = classifierType[z2]

                maxAcc = 0
                #scoreBestPCs = numpy.zeros(scoreTotal.shape)
                scoreBestPCs = []
                bestPCIndex = numpy.array([])


                PCNoTOTAL = PCNumTOTAL
                scoreTOTAL = scoreTotal
                pcIndexNo = pcIndexNumbers

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
                    print("in the while loop")
                    if lt > 1:

                        if scoreBestPCs != []:
                            scoreBestPCs = numpy.column_stack((scoreBestPCs, scoreTOTAL[:, maxAccIndex]))
                        else:
                            scoreBestPCs = scoreTOTAL[:, maxAccIndex]
                        if bestPCIndex.size != 0:
                            bestPCIndex = numpy.append(bestPCIndex, [pcIndexNo[maxAccIndex]])
                            #bestPCIndex = numpy.append((bestPCIndex, pcIndexNo[maxAccIndex]))
                        else:
                            bestPCIndex = pcIndexNo[maxAccIndex]

                        #should be 0?
                        if maxAccIndex == 1:
                            scoreTOTAL = scoreTOTAL[:, 1:PCNoTOTAL]
                            pcIndexNo = pcIndexNo[1:PCNoTOTAL]

                        elif maxAccIndex == PCNoTOTAL-1:
                            scoreTOTAL = scoreTOTAL[:, 0:PCNoTOTAL - 1]
                            pcIndexNo = pcIndexNo[0:PCNoTOTAL - 1]
                        else:
                            scoreTOTAL = numpy.column_stack((scoreTOTAL[:, 0:maxAccIndex],
                                                            scoreTOTAL[:, maxAccIndex+1:PCNoTOTAL]))
                            pcIndexNo = numpy.row_stack((pcIndexNo[0:maxAccIndex],
                                                           pcIndexNo[maxAccIndex+1:PCNoTOTAL]))

                        lowClassAccuracy = lowClassAccuracies[0][maxAccIndex]
                        highClassAccuracy = highClassAccuracies[0][maxAccIndex]

                        finalInstMisclass = instMisclass[maxAccIndex]

                        print("finalInstMisclass:")
                        print(finalInstMisclass)

                        #numpy function for row concatenation
                        if accIncrTracker.size != 0:
                            accIncrTracker = numpy.append(accIncrTracker, [(maxAccTracker[1] - maxAccTracker[0])])
                        else:
                            accIncrTracker = maxAccTracker[1]-maxAccTracker[0]

                        maxAccuracy = maxAcc

                        PCNoTOTAL = PCNoTOTAL - 1
                        #end not checked

                    accuracies = numpy.zeros((1, PCNoTOTAL))
                    lowClassAccuracies = numpy.zeros((1, PCNoTOTAL))
                    highClassAccuracies = numpy.zeros((1, PCNoTOTAL))
                    #instMisclass = numpy.zeros((1, PCNoTOTAL))
                    instMisclass = [0] * PCNoTOTAL

                    for i in range(0, PCNoTOTAL):
                        #numpy function for column concatenation
                        #print(scoreBestPCs.shape)
                        #print(scoreTotal.shape)
                        scoreCandidatePCs = 0
                        if scoreBestPCs != []:
                            scoreCandidatePCs = numpy.column_stack((scoreBestPCs, scoreTOTAL[:, i]))
                        else:
                            scoreCandidatePCs = numpy.reshape(scoreTotal[:, i], (len(scoreTotal), 1))

                        preAccMatrix = numpy.zeros((len(scoreCandidatePCs), 3))
                        preInstOrder = numpy.zeros((len(scoreCandidatePCs), 1))

                        #x = 0     put in classifierTrainTest
                            #FIX: lines 280-285

                        #for j in range(0, foldNo):        put loop in classifierTrainTest
                        if cvStatus == 0:
                            USFS.classifierTrainTest(scoreCandidatePCs, response_data, real_art_data, cvPartition, classifier,
                                                instanceIndex, preAccMatrix, preInstOrder)
                            real_artTEST = dict.get('real_artTEST')
                            instIndexTEST = dict.get('instIndexTEST')
                            trueClassLabel = dict.get('trueClassLabel')
                            predictedClassLabel = dict.get('predictedClassLabel')
                            #return all of idx[j] to idx[z2][z1]
                            idx[z2][z1][j] = dict.get('idx')

                        else:
                            dict = USFS.classifierTrainTest(scoreCandidatePCs, response_data, real_art_data, cvPartition, classifier,
                                                  instanceIndex, preAccMatrix, preInstOrder)
                            real_artTEST = dict.get('real_artTEST')
                            instIndexTEST = dict.get('instIndexTEST')
                            trueClassLabel = dict.get('trueClassLabel')
                            predictedClassLabel = dict.get('predictedClassLabel')

                            subAccMatrix = dict.get('subAccMatrix')
                            preAccMatrix = dict.get('preAccMatrix')
                            preInstOrder = dict.get('preInstOrder')


                            #Added these lines to classifierTrainTest
                            #subAccMatrix = numpy.column_stack(trueClassLabel, predictedClassLabel, real_artTEST)
                            #preAccMatrix[x:x + len(subAccMatrix[:, 0]) - 1, :] = subAccMatrix
                            #preInstOrder[x:x + len(instIndexTEST[:, 0]) - 1] = instIndexTEST

                            #x = x + (subAccMatrix[:, 0].size)

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
                        # FIX: line 313
                        dict2 = USFS.accuracyCalculation(accMatrix, lowClass, instOrder)
                        accuracies[0][i] = dict2.get('accuracy')
                        lowClassAccuracies[0][i] = dict2.get('lowClassAccuracy')
                        highClassAccuracies[0][i] = dict2.get('highClassAccuracy')
                        instMisclass[i] = dict2.get('instMisclass')

                        # FIX: line 318

                    maxAccIndex = numpy.argmax(accuracies)
                    maxAcc = numpy.amax(accuracies)

                    if (maxAccTracker[0] == 0) and (maxAccTracker[1] == 100):
                        maxAccTracker = numpy.array([0, maxAcc])
                    else:
                        maxAccTracker[0] = maxAccTracker[1]
                        maxAccTracker[1] = maxAcc

                    if (PCNoTOTAL == 1) and ((maxAccTracker[1] - maxAccTracker[0]) > 1):
                        scoreBestPCs = numpy.column_stack((scoreBestPCs, scoreTOTAL))
                        bestPCIndex = numpy.hstack((bestPCIndex, pcIndexNo))

                        scoreTOTAL = []
                        pcIndexNo = []

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
                bestPCS[z2][z1] = bestPCIndex
                accIncr[z2][z1] = accIncrTracker
                # FIX: 3d array where you can replace a whole column
                shapeSubjMisclassified = subjMisclassified[z2].shape
                for c in range(0, shapeSubjMisclassified[0]):
                    subjMisclassified[z2][c][1 + z1] = finalInstMisclass[c]
                #################################################

        maxVal = numpy.zeros(classifNo)
        for i in range(0, classifNo):
            for j in range(0, iterationLength):
                bestPCsShape = bestPCS[i][j].shape
                if bestPCsShape[0] > maxVal[i]:
                    maxVal[i] = bestPCsShape[0]

        bestPCsummary = [0]*classifNo

        # FIX: curly brackets vs parentheses? lines 387-391
        for i in range(0, classifNo):
            bestPCsummary[i] = numpy.zeros((3 + maxVal[i], iterationLength * 2))
            bestPCsummary[i][0, 0: iterationLength] = accuracyOverall[i]
            bestPCsummary[i][1, 0: iterationLength] = accuracyFirstClass[i]
            bestPCsummary[i][2, 0: iterationLength] = accuracySecondClass[i]

        for i in range(0, iterationLength):
            for j in range(0, classifNo):
                x = 3
                bestPCsShape = bestPCS[j][i].shape
                for k in range(0, bestPCsShape[0]):
                    bestPCsummary[j][x][i] = bestPCS[j][i][k]
                    bestPCsummary[j][x][i + iterationLength] = accIncr[j][i][k]
                    x += 1

        for i in range(0, classifNo):
            summary_string = 'SummaryBestPCS_' + classifierType[i] + '.csv'
            misclassified_string = 'MisclassifiedSubjects_' + classifierType[i] + '.csv'
            file3 = open(summary_string, 'wb')
            file4 = open(misclassified_string, 'wb')
            numpy.savetxt(file3, bestPCsummary[i], delimiter=',')
            numpy.savetxt(file4, subjMisclassified[i], delimiter=',')
            file3.close()
            file4.close()

        #return {'classifierType': classifierType, '': }
            #pcsummaryShape = bestPCsummary[i].shape
            #for q in range(0, pcsummaryShape[0]):
            #    wr3.writerow(bytes(bestPCsummary[i][q], 'UTF-8'))


            #file4 = open(misclassified_string, 'wb')
            #wr4 = csv.writer(file4, quoting=csv.QUOTE_ALL)
            #submisclshape = subjMisclassified[i].shape
            #for q in range(0, submisclshape[0]):
            #    wr4.writerow(bytes(subjMisclassified[i][q], 'UTF-8'))

            #file3.close()
            #file4.close()


    # for Monday: fix syntax and compiler errors, compile code using leave 1 out cross validation (run as is)
    # install packages
    # email Dr. Wu for thesis approval --done


    #USFS()























