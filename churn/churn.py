
#autoscaling, centering
#0-s?
#sq 
# decision trees, Bayesian networks, support vector machines, and neural networks.
# the effort to reduce one type of error generally results in increasing the other type of error
#PPV NPV
#the sensitivity of the test can be determined by testing only positive cases

# StandardScaler OneHotEncoder

#import sys
#sys.path.insert(0,'/home/rewlad/tmp/lib/python')
#print sys.path

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import sklearn.preprocessing as sp
import sklearn.decomposition as sk
import sklearn.metrics as sm

def load_rows():
    with open('Churn.csv','rb') as csvfile:
        reader = csv.reader(csvfile,delimiter=',',quotechar='"')
        rows = []
        for rn,row in enumerate(reader):
            if rn==0: head = row[0:22]
            else: rows.append(row[0:22])
        return rows, head

def rows_to_predictor_response(rows, head):
    codes = [row[19] for row in rows]
    codes_h = { v:j for j,v in enumerate(set(codes)) }
    for row in rows:
        for cn, v in enumerate(row):
            row[cn] = float(v) if cn<18 else codes_h[v] if cn==19 else 0
    arows = np.array(rows)
    x_row_indexes = [0,1,2,3,4,5,6, 8,9,10,11,12,13,14,15,16,17, 19]
    x_head = np.array(head)[x_row_indexes]
    X = arows[:, x_row_indexes]
    Y = arows[:, [7]]
    return X, Y, x_head
    
def mode_check_rows():
    rows, head = load_rows()
    print x_head
    phones = [row[20] for row in rows]
    print len(phones), len(set(phones)) #checking phones are uniq
    codes = [row[19] for row in rows]
    states = [row[18] for row in rows]
    print set(codes), set(states) #codes '415', '510', '408' all from California
    
    X, Y, x_head = rows_to_predictor_response(rows, head)
    print head, x_head, head[7]
    print X.shape, Y.shape

def mode_main():
    rows, head = load_rows()
    X, Y, x_head = rows_to_predictor_response(rows, head)
    
    train = np.arange(len(X)) > len(X)/4
    test  = np.arange(len(X)) <= len(X)/4

    X_train = X[train]
    Y_train = Y[train]

    x_mean = X_train.mean(axis=0)
    X_c = X_train - x_mean
    x_std = X_c.std(axis=0)
    X_n = X_c / x_std

    print x_mean, x_std

    pca = sk.PCA(n_components=10)
    pc = pca.fit_transform(X_n)

    churn = Y_train[:,0] > 0.5

    for i in range(1,9):
        plt.plot(pc[:,0], pc[:,i], 'go')
        plt.plot(pc[churn,0], pc[churn,i], 'ro')
        plt.savefig("churn_scores_0_"+str(i)+".png")
        plt.cla()

    loadings = pca.components_

    plt.plot(loadings[0], loadings[1], 'go')
    for i,l in enumerate(loadings[0:2].T): 
        plt.annotate(x_head[i],l+(i%3)*0.02);
        print x_head[i],l
    plt.savefig("churn_loadings.png")
    plt.cla()
    """
    y_mean = Y_train.mean()
    Y_c = Y_train-y_mean
    y_std = Y_c.std()
    Y_n = Y_c / y_std
    """

    y_mean = Y_train.mean()
    Y_c = Y_train-y_mean
    y_std = 1
    Y_n = Y_c
    """
    y_mean = 0
    y_std = 1
    Y_n = Y_train
    """

    (a,residues,rank,s) = la.lstsq(pc,Y_n)

    pc_test = pca.transform((X[test]-x_mean)/x_std)

    Y_pred = pc_test.dot(a) * y_std + y_mean

    for p in np.hstack(( np.round(Y_pred,decimals=2), Y[test] )): print p


