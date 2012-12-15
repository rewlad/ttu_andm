
#autoscaling, centering
#0-s?
#sq 
# decision trees, Bayesian networks, support vector machines, and neural networks.
# the effort to reduce one type of error generally results in increasing the other type of error
#PPV NPV
#the sensitivity of the test can be determined by testing only positive cases

# StandardScaler OneHotEncoder
# LogisticRegression

#import sys
#sys.path.insert(0,'/home/rewlad/tmp/lib/python')
#print sys.path

import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import sklearn.pipeline as spl
import sklearn.preprocessing as spp
import sklearn.decomposition as sk
import sklearn.linear_model as slm
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

    pca = sk.PCA(n_components=10)
    pipeline = spl.Pipeline([ ('scaler',spp.Scaler()), ('pca',pca) ])
    
    pc = pipeline.fit_transform(X[train])
    """
    churn = Y[train,0] > 0.5
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
    y_scaler = spp.Scaler(with_std=False)
    linre = slm.LinearRegression()
    linre.fit(X=pc, y=y_scaler.fit_transform(Y[train,0]))
    
    Y_pred = y_scaler.inverse_transform( linre.predict( pipeline.transform(X[test]) ) )
    
    #for p in np.hstack(( np.round(Y_pred,decimals=2)[:,None], Y[test] )): print p
    fpr, tpr, thresholds = sm.roc_curve(y_true=Y[test,0], y_score=Y_pred)
    print sm.auc(fpr, tpr)
    #print sm.classification_report(y_true=Y[test,0], y_pred=Y_pred)
    plt.plot(fpr, tpr, 'g-')
    plt.savefig("roc.png")
mode_main()

"""
linre = slm.LinearRegression()
linre.fit(
    X=np.array([
        [0,1,2],
        [2,3,4],
        [4,5,6]
    ]),
    y=np.array([
        6,
        7,
        8
    ]),
)
print linre.predict(
    X=np.array([
        [6,7,8],
        [8,9,10],
    ])
)
"""

