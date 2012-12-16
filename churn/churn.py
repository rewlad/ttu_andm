
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

import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn import pipeline, preprocessing, decomposition, linear_model, pls, tree, metrics, utils

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
    arows = utils.shuffle(np.array(rows), random_state=0)
    x_row_indexes = [0,1,2,3,4,5,6, 8,9,10,11,12,13,14,15,16,17, 19]
    x_head = np.array(head)[x_row_indexes]
    X = arows[:, x_row_indexes]
    Y = arows[:, [7]]
    rng, l = np.arange(len(X)), len(X)/3
    train, test = rng > l, rng <= l
    return X, Y, x_head, train, test
    
def mode_check_rows():
    rows, head = load_rows()
    print x_head
    phones = [row[20] for row in rows]
    print len(phones), len(set(phones)) #checking phones are uniq
    codes = [row[19] for row in rows]
    states = [row[18] for row in rows]
    print set(codes), set(states) #codes '415', '510', '408' all from California
    
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    print head, x_head, head[7]
    print X.shape, Y.shape
    print len(Y[train]), len(Y[test]), np.count_nonzero(Y[train]), np.count_nonzero(Y[test])
    print np.count_nonzero(Y)/float(len(Y))
    print np.count_nonzero(Y[train])/float(len(Y[train]))
    print np.count_nonzero(Y[test])/float(len(Y[test]))
    
def plot_rocs(nm,color,y_true,y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true=y_true, y_score=y_pred)
    print metrics.auc(fpr, tpr)
    if len(fpr)<4: print fpr, tpr
    path = "roc."+nm+".npz"
    np.savez(path,fpr=fpr, tpr=tpr, thresholds=thresholds, color=np.array([color]))
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    for fn in glob.glob("roc.*.npz"): 
        v = np.load(fn)
        plt.plot(v['fpr'], v['tpr'], v['color'][0])
    plt.savefig("roc.png")
    plt.cla()
    
    
def mode_pca():
    rows, head = load_rows()
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    
    pca = decomposition.PCA(n_components=13)
    re_pipeline = pipeline.Pipeline([ ('scaler',preprocessing.Scaler()), ('pca',pca) ])
    pc = re_pipeline.fit_transform(X[train])
    
    churn = Y[train,0] > 0.5
    for i in range(1,13):
        plt.plot(pc[:,0], pc[:,i], 'go')
        plt.plot(pc[churn,0], pc[churn,i], 'ro')
        plt.savefig("churn_scores_0_"+str(i)+".png")
        plt.cla()

    loadings = pca.components_
    for i in range(1,13):
        plt.plot(loadings[0], loadings[i], 'go')
        for j,l in enumerate(loadings[[0,i]].T): 
            plt.annotate(x_head[j],l); #-(j%3)*0.02
            #print x_head[i],l
        plt.savefig("churn_loadings_0_"+str(i)+".png")
        plt.cla()
    
    y_scaler = preprocessing.Scaler(with_std=False)
    linre = linear_model.LinearRegression()
    linre.fit(X=pc, y=y_scaler.fit_transform(Y[train,0]))
   
    y_pred = y_scaler.inverse_transform( linre.predict( re_pipeline.transform(X[test]) ) )
    
    plot_rocs('pca','b-',Y[test,0],y_pred)

def mode_pls():
    rows, head = load_rows()
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    
    re = pls.PLSRegression(n_components=11)
    re.fit(X=X[train], Y=Y[train])
    y_pred = re.predict( X[test] )[:,0]
    
    plot_rocs('pls','b-',Y[test,0],y_pred)

def mode_linre():
    rows, head = load_rows()
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    
    linre = linear_model.LinearRegression()
    linre.fit(X=X[train], y=Y[train,0])
    y_pred = linre.predict( X[test] )
    
    plot_rocs('linre','b-',Y[test,0],y_pred)
    
def mode_tree():
    rows, head = load_rows()
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    
    clf = tree.DecisionTreeClassifier(max_depth=5)#,min_samples_leaf?
    clf.fit(X=X[train], y=Y[train,0]) #*2-1
    #y_pred = clf.predict( X[test] )
    y_pred = clf.predict_proba( X[test] )[:,1]
    
    plot_rocs('tree','g-',Y[test,0],y_pred)
    
    tree.export_graphviz(clf, out_file='dtree.graphviz',feature_names=x_head)
    #convert using: dot dtree.graphviz -Tpng > dtree2.png

def mode_pca_tree():
    rows, head = load_rows()
    X, Y, x_head, train, test = rows_to_predictor_response(rows, head)
    
    pca = decomposition.PCA(n_components=13)
    re_pipeline = pipeline.Pipeline([ ('scaler',preprocessing.Scaler()), ('pca',pca) ])
    pc = re_pipeline.fit_transform(X[train])
    
    clf = tree.DecisionTreeClassifier(max_depth=5)
    clf.fit(X=pc, y=Y[train,0])
    y_pred = clf.predict( re_pipeline.transform(X[test]) )
    
    plot_rocs('pca_tree','r-',Y[test,0],y_pred)
    
    tree.export_graphviz(clf, out_file='pca_dtree.graphviz',feature_names=x_head)
    

mode_linre()
mode_pca()
mode_pls()
mode_tree()
#mode_pca_tree()
"""
clf = tree.DecisionTreeClassifier()
clf.fit(
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
tree.export_graphviz(clf, out_file='test.graphviz')
"""
"""
linre = linear_model.LinearRegression()
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
#y_repred = clf.predict(X[train]);
#print metrics.roc_curve(y_true=Y[train,0], y_score=y_repred) #fpr, tpr
    
#print metrics.classification_report(y_true=Y[test,0], y_pred=Y_pred)
#for p in np.hstack(( np.round(Y_pred,decimals=2)[:,None], Y[test] )): print p

################################################################################


#LogisticRegression gives random 0.5 :-(
#linre = linear_model.LogisticRegression(C=100000.0, dual=False, fit_intercept=True, intercept_scaling=1, penalty='l2', tol=0.0001)

#0.5
#re = svm.SVC()    re.fit(X[train], Y[train,0])    y_pred = re.predict( X[test] )

#with 13 pc-s same 0.83- as without it, less worse
#pca = decomposition.PCA(n_components=13)

#pls -- same 0.83+

#DecisionTreeClassifier 0.86/0.82 -- a bit better
#0.03; 0.7

#found clusters at pc3: vmail+night

#pca then tree: same or worse


#try clusters at pc3
#try? find deps from orig vars
#try preproc nonlinear
#try multi vars for codes
#try tree, if fails then other

#try by wiki: Bayesian networks and neural networks.



