
import csv
from numpy import array

def to_float(v):
    print "["+v+"]"
    return 0. if v=='' else 1. if v=='X' else float(v)

with open('Bertin_s_hotel_example.csv','rb') as csvfile:
    reader = csv.reader(csvfile,delimiter=',',quotechar='"')
    rows = array([[to_float(cell) for cn,cell in enumerate(row) if cn>1] for rn,row in enumerate(reader) if rn>0]).T
    #rows_pure = rows[1:,2:]

    print rows

    #for row in rows:
    #    print '| '.join(str(row))