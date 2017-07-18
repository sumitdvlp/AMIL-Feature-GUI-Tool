import Go
import CFS
import numpy as np
from sklearn.cross_validation import train_test_split
from skfeature.function.similarity_based import fisher_score
from sklearn import svm
from sklearn.metrics import accuracy_score
import trace_ratio
from sklearn.linear_model import LogisticRegression
from scipy import stats
import statsmodels
import reliefF
import MIM
import matplotlib.pyplot as plt
import re
import pandas as pd
import sklearn
import os
import numpy as np
import csv
from skfeature.utility.entropy_estimators import *
from skfeature.function.information_theoretical_based import MIFS, ICAP
from skfeature.function.information_theoretical_based import CIFE
from skfeature.function.information_theoretical_based import JMI
from skfeature.function.information_theoretical_based import FCBF
from skfeature.function.information_theoretical_based import LCSI
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CMIM
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.sparse_learning_based import *
from skfeature.function.statistical_based import t_score
from skfeature.function.statistical_based import gini_index
from skfeature.function.statistical_based import chi_square
from sklearn.preprocessing import StandardScaler
from skfeature.function.information_theoretical_based import MIM
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from skfeature.function.statistical_based import f_score
from skfeature.function.statistical_based import low_variance
from skfeature.function.statistical_based import CFS
from skfeature.function.statistical_based import t_score
from skfeature.function.sparse_learning_based import RFS
from skfeature.function.wrapper.svm_forward import svm_forward

class FeatureSelect():
    X_train = []
    num_fea = 2
    X_test = []
    idx = []
    Fe1 = []
    Fe2 = []
    F1 = []
    F2 = []
    X=[]
    y=[]
    featureName=[]

    def __init__(self,path):
        self.read_Data_Method2()

    def read_Data_Method3(self):
        myGo = Go.Go()
        response = "/home/launch/Desktop/Share/response.csv"
        real_art = "/home/launch/Desktop/Share/real_art.csv"
        data = "/home/launch/Desktop/Share/data.zip"
        processedFiles = myGo.processFiles(response, real_art, data)
        self.X = processedFiles["data"]
        self.y = processedFiles["response"]
    def read_Data_Method2(self):
        xpath='/home/launch/Desktop/Share/data49 with name/X'
        ypath='/home/launch/Desktop/Share/data49 with name/y'

        merged =[]
        data=pd.DataFrame()
        for file in os.listdir(xpath):
            filePath=xpath+'/'+file
            ydata=pd.read_csv(filePath,index_col=None,header=0)
            merged.append(ydata)

        data=pd.concat(merged,axis=1)
        columnName=data.columns.values

        self.X = data.as_matrix()
        self.featureName=columnName.tolist()

        file=os.listdir(ypath)
        filePath = ypath + '/' + file[0]
        ydata = pd.read_csv(filePath, index_col=None, header=0)

        #print(ydata.values)
        self.y=ydata.iloc[:, 0].tolist()

        '''with open(filePath, 'r') as my_file:
            reader = csv.reader(my_file, delimiter=',')
            my_list = list(reader)
            print(type(my_list))
        self.y=my_list'''

    def read_Data_Method1(self):
        pdata = pd.read_csv("/media/sf_Share/CEDM_51_Updated_features.csv", header=None)
        df = pd.DataFrame(pdata)

        df_merged = pd.DataFrame()
        for name, group in df.groupby(0):
            df_merged = df_merged.append(pd.DataFrame(group.values[:, 1:].reshape(1, -1)))

        X = df_merged.as_matrix(columns=df_merged.columns[2:])
        print(type(X))

        pdatay = pd.read_csv("/media/sf_Share/y.csv", header=None)
        dfy = pd.DataFrame(pdatay)

        df_mergedy = pd.DataFrame()
        for name, group in dfy.groupby(0):
            df_mergedy = df_mergedy.append(pd.DataFrame(group.values[:, 1:].reshape(1, -1)))

        y = df_mergedy.as_matrix(columns=df_mergedy.columns[1:2])
        print(type(y))
        #np.savetxt("/media/sf_Share/X.csv", X, delimiter=',')
        #np.savetxt("/media/sf_Share/yres.csv", y, delimiter=',')

    def __init__(self, *args, **kwargs):
        num_fea = 2
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=40)


    def svm_forward(self,n):
        F=svm_forward(self.X,self.y,n)
        return F

    def JMI(self):
        idx = JMI.jmi(self.X, self.y, n_selected_features=2)
        #selected_features_train = self.X_train[:, idx[0:self.num_fea]]
        #selected_features_test = self.X_test[:, idx[0:self.num_fea]]
        self.F1 = self.X[:, idx.item(0)]
        self.F2 = self.X[:, idx.item(1)]
        ttestt = stats.ttest_ind(self.F1, self.F2)
        return idx,ttestt

    def F_SCORE(self):

        f = f_score.f_score(self.X, self.y)
        idx=f_score.feature_ranking(f)
        self.F1 = self.X[:, idx.item(0)]
        self.F2 = self.X[:, idx.item(1)]
        ttestt = stats.ttest_ind(self.F1, self.F2)
        return idx

    def get_MIM(self,n):
        print(self.y)
        print(type(self.y))
        idx=MIM.mim(self.X,self.y)
        return idx

    def CMIM(self):
        idx = MIM.mim(self.X, self.y, n_selected_features=2)
        self.F1 = self.X[:, idx.item(0)]
        self.F2 = self.X[:, idx.item(1)]
        ttestt = stats.ttest_ind(self.F1, self.F2)
        return idx,ttestt

    def get_values(self,idx):
        self.F1 = self.X[:, idx[0]]
        self.F2 = self.X[:, idx[1]]

        F10 = []
        F20 = []
        F11 = []
        F21 = []
        for i in range(len(self.y)):
            if (self.y[i] == 0):
                F10.append(self.F1[i])
                F20.append(self.F2[i])
            else:
                F11.append(self.F1[i])
                F21.append(self.F2[i])

        self.Fe1 = []
        self.Fe1.append(F10)
        self.Fe1.append(F11)

        self.Fe2 = []
        self.Fe2.append(F20)
        self.Fe2.append(F21)
        return self.Fe1,self.Fe2

    def get_Feature_List(self,idx,nFea):
        ret=[]
        featName=[]
        for n in range(0,nFea):
            featName.append(self.featureName[idx[n]])

        print(featName)

        for i in range(0,nFea):
            Feat=self.X[:,idx[i]]
            Fe10=[]
            Fe11=[]
            for j in range(len(self.y)):
                if(self.y[j]==0):
                    Fe10.append(Feat[j])
                else:
                    Fe11.append(Feat[j])
            Fe1=[]
            Fe1.append(Fe10)
            Fe1.append(Fe11)
            Fe10=MinMaxScaler().fit_transform(Fe10)
            Fe11=MinMaxScaler().fit_transform(Fe11)
            ttest=ascii(stats.ttest_ind(Fe10,Fe11))
            m = re.split(r'pvalue=', ttest)
            ttestVal = re.sub(r'\)', '', m[1])

            ttestVal=round(float(ttestVal),4)
            ReFe1=[]
            ReFe1.append(Fe1)
            ReFe1.append(ttestVal)
            ret.append(ReFe1)
        return ret,featName

    def get_scaled_values(self,idx):
#        self.F1 = self.X[:, idx.item(0)]
#        self.F2 = self.X[:, idx.item(1)]
        self.F1 = self.X[:, idx[0]]
        self.F2 = self.X[:, idx[1]]
        F10 = []
        F20 = []
        F11 = []
        F21 = []
        for i in range(len(self.y)):
            if (self.y[i] == 0):
                F10.append(self.F1[i])
                F20.append(self.F2[i])
            else:
                F11.append(self.F1[i])
                F21.append(self.F2[i])

        F10= MinMaxScaler().fit_transform(F10)
        F20= MinMaxScaler().fit_transform(F20)
        F11 = MinMaxScaler().fit_transform(F11)
        F21 = MinMaxScaler().fit_transform(F21)
        F10.tolist()
        F11.tolist()
        F21.tolist()
        F20.tolist()
        return F10,F20,F11,F21
    def get_ttest(self):
        idx = self.F_SCORE()
        Fs10, Fs20,Fs11,Fs21 = self.get_scaled_values(idx)
        ttest1=stats.ttest_ind(Fs10,Fs11)
        ttest2=stats.ttest_ind(Fs20,Fs21)
        return ttest1,ttest2

    def draw(self):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))
        axes[0].violinplot(self.Fe1, showmeans=False, showmedians=True)
        axes[0].set_title('Feature 1')

        axes[1].violinplot(self.Fe2, showmeans=False, showmedians=True)
        axes[1].set_title('Feature 2')
        plt.setp(axes, xticks=[y + 1 for y in range(len(self.Fe1))], xticklabels=['Y=0', 'y=1'])
        plt.show()


# arr=np.asarray(F1)
# print(type(arr))
# min_max_scaler = preprocessing.MinMaxScaler()
# X = MinMaxScaler().fit_transform(arr)
# print(X)
'''
score = fisher_score.fisher_score(X_train, y_train)
idx = fisher_score.feature_ranking(score)
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

rvs1=selected_features_train[:,0]
rvs2=selected_features_train[:,1]
ttestt=stats.ttest_ind(rvs1,rvs2)
print('ttestt',ttestt)
clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
print('y_predict',y_predict)
rvs1=selected_features_test[:,0]
print('rvs ',rvs1)
rvs2=selected_features_test[:,1]
print('rvs ',rvs2)
acc = accuracy_score(y_test, y_predict)
print (acc)
'''

'''
varKept = myGo.DimReduction(float(85), processedFiles["response"],processedFiles["dataList"])
#fs = myGo.runLASSO(varKept["scoreTotal"], processedFiles["response"])
#fs = myGo.Lasso_new(varKept["scoreTotal"], processedFiles["response"])
#print('fs is', fs)
X=np.asarray(varKept["scoreTotal"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

score = fisher_score.fisher_score(X_train, y_train)
idx = fisher_score.feature_ranking(score)

num_fea = 25
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print (acc)'''

##trace_ratio acuracy same as fisher as it runs fisher in background
# if use laplase it its taking alot of time while converging
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
num_fea = 5
kwargs={'style':'laplacian','verbose':'True'}
idx,score,subset_score = trace_ratio.trace_ratio(X_train, y_train,num_fea,style='laplacian',verbose='False')
idx,score,subset_score = trace_ratio.trace_ratio(X_train, y_train,num_fea,style='laplacian')
selected_features_train = X_train[:, idx]
selected_features_test = X_test[:, idx]
clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print('LR-',acc)
'''

## tried releiF , its giving TypeError: object of type 'float' has no len()
## maybe 2.7 to 3 error
'''X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)
num_fea = 5
score = reliefF.reliefF(X_train, y_train)
print(score)
'''

### MIM
## tried - error TypeError: object of type 'zip' has no len()
## same error with CFS i think
# idx = MIM.mim(X_train, y_train,n_selected_features=5)
# print(idx)



##MIFS
## tried - error TypeError: object of type 'zip' has no len()
## same error with CFS i think
# idx =MIFS.mifs(X_train, y_train,n_selected_features=5)
# print(idx)

##CIFE
## tried - error TypeError: object of type 'zip' has no len()
## same error with CFS i think
# idx =CIFE.cife(X_train, y_train,n_selected_features=5)
# print(idx)

##JMI

##MRMR
# idx =MRMR.mrmr(X_train, y_train,n_selected_features=5)
# print(idx)


##so many error
##moving to statitical based learning
# Gini Square
'''
gini = gini_index.gini_index(X_train, y_train)
idx = t_score.feature_ranking(gini)
print(idx)
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print ('tscore',acc)
'''

'''
#chi_Square
F = chi_square.chi_square(X_train, y_train)
idx = chi_square.feature_ranking(F)
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print ('chi_square',acc)'''

'''
#F-SCORE
F = f_score.f_score(X_train, y_train)
idx = f_score.feature_ranking(F)
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print ('f_score',acc)

#tscore
F = t_score.t_score(X_train, y_train)
idx = t_score.feature_ranking(F)

selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print ('t_score--',acc)
'''
'''
#cfs
idx = CFS.cfs(X_train, y_train)
selected_features_train = X_train[:, idx[0:num_fea]]
selected_features_test = X_test[:, idx[0:num_fea]]

clf = LogisticRegression()
clf.fit(selected_features_train, y_train)
y_predict = clf.predict(selected_features_test)
acc = accuracy_score(y_test, y_predict)
print ('cfs--',acc)

#sparse_learning_based.RFS
idx = RFS.rfs(X_train, y_train)
print(idx)
'''
