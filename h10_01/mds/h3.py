
import csv
from numpy import array, zeros,where,random,ones,sqrt

def to_float(v):
    print "["+v+"]"
    return 0. if v=='' else 1. if v=='X' else float(v)

with open('mdsAlgus2012.csv','rb') as csvfile:
    reader = csv.reader(csvfile,delimiter=',',quotechar='"')
    rows = []
    for rn,row in enumerate(reader):
        if rn==0: head = row[1:28]
	else: rows.append(row[1:28])
rows = array(rows).T
brows = where(rows != '0',1,0)
hdist = zeros((27,27))
for cn in range(27):
    hdist[cn] = (brows != brows[:,cn]).sum(axis=1)

points = random.random((27,2)) * 100

def F(points):
    x0 = points[:,0:1].dot(ones((1,27)))
    y0 = points[:,1:2].dot(ones((1,27)))
    dist = sqrt((x0-x0.T)**2+(y0-y0.T)**2)
    return (dist-hdist).sum()

#print f(points)

import scipy.optimize
x = scipy.optimize.broyden2(F, points)
print x


#(x0-x1)^2 + (y0-y1)^2
#print points, dist#points[:,0].repeat(2,axis=0)

    #print (brows[:,0]  != brows) .sum(1)  
#print  hdist
#print array([1,2]) != array([[1,3],[1,2]])

    #for row in rows:
    #    print '| '.join(str(row))