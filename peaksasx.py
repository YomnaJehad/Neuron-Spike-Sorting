
##TO DRAW X'S ON THE PEAKS DETECTED
import numpy as np
import matplotlib.pyplot as plt


peaks=np.load('/media/yomnaj/New Volume/Bio_Assignment_2/electrode1thresh35peaksss.txt.npy')

peaksIndex=np.load('/media/yomnaj/New Volume/Bio_Assignment_2/electrode1thresh35peaksssIndex.txt.npy')


fileName="Data.txt"
#electrode1,electrode2 = np.loadtxt(filename, delimiter=" ")
def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y

electrode1,electrode2 = Read_Two_Column_File(fileName)
samplingRate= 24414

x_values =  range(0,len(electrode1))

plt.plot(x_values, electrode1 )
plt.plot(peaksIndex, peaks, 'x' )
plt.show()