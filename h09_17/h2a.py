
import csv
import numpy as np
import matplotlib.pyplot as plt

def to_float(v,rn):
    return 0. if v=='' or rn==0 else 1. if v=='X' else float(v)

def save(nm):
    plt.savefig("out/"+nm+".png")
    plt.cla()

with open('Bertin_s_hotel_example.csv','rb') as csvfile:
    reader = csv.reader(csvfile,delimiter=',',quotechar='"')
    rows = np.array([
        [to_float(cell,rn) for cn,cell in enumerate(row) if cn>1] 
        for rn,row in enumerate(reader)
    ])
    
    room_price = rows[17]
    occupancy = rows[19]
    profit = room_price * occupancy
    conventions = rows[20]
    
    for rn in range(1,rows.shape[0]):
        plt.plot(rows[rn],profit,'go')
        save("f"+str(rn)+"_pro")
    
    plt.plot(room_price,occupancy,'go')
    save("pri_occ")
    
    plt.plot(range(12),occupancy,'go')
    save("mon_occ")
    
    plt.plot(range(12),room_price,'go')
    plt.plot(range(12),conventions*200,'bo')
    save("mon_pri_con")

