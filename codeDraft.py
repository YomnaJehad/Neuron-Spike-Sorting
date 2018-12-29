#imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from scipy.ndimage.interpolation import shift

#____________________________________________________________________________________________________________
# read the data file
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
#____________________________________________________________________________________________________________
#Let's visualize the electrodes readings 
print('length of electrode1 list',len(electrode1))
print('length of electrode2 list',len(electrode2))
#____________________________________________________________________________________________________________
x_values = range(0,len(electrode1))
#plt.plot(x_values,electrode1)
#plt.plot(x_values,electrode2)

"""

f1, (ax1,ax2)=plt.subplots(1,2,figsize=(24,9))
f1.tight_layout()

ax1.plot(x_values,electrode1)
ax1.set_title('electrode1',fontsize=30)

ax2.plot(x_values,electrode2)
ax2.set_title('electrode2',fontsize=30)
"""
#____________________________________________________________________________________________________________

electrode1Abs= np.absolute(electrode1)
electrode2Abs= np.absolute(electrode2)

"""
f2, (ax3,ax4)=plt.subplots(1,2,figsize=(24,9))
f2.tight_layout()

ax3.plot(x_values,electrode1)
ax3.set_title('electrode1 absolute',fontsize=30)

ax4.plot(x_values,electrode2)
ax4.set_title('electrode2 absolute',fontsize=30)
"""
#____________________________________________________________________________________________________________
sd1 = np.std(electrode1Abs[:500])
sd2 = np.std(electrode2Abs[:500])
print('SD',sd1,sd2)

thresh1, thresh2=sd1*3.5 ,sd2*3.5
print('THRESHOLDS ',thresh1, thresh2)

#____________________________________________________________________________________________________________

def smoothing(squares,window_size=48):
    
    N=window_size
    smooth=np.zeros((squares.shape[0]))
    for i in range(N//2,(squares.shape[0]-(N//2))):
        
        sum=0
        for j in range (i-(N//2), i+(N//2)):
            
            sum=sum+squares[j]
        smooth[i]=(1/N)*sum
    return smooth

electrode1Smooth= smoothing(electrode1Abs,48)
electrode2Smooth=  smoothing(electrode2Abs,48)
print(electrode1Smooth.shape)
# f2, (ax3,ax4)=plt.subplots(1,2,figsize=(24,9))
# f2.tight_layout()

# ax3.plot(x_values,smoothedElectrode1)
# ax3.set_title('electrode1',fontsize=30)

# ax4.plot(x_values,smoothedElectrode2)
# ax4.set_title('electrode2',fontsize=30)




#____________________________________________________________________________________________________________










def getSpikes(electrodeAbs , electrode,thresh , noSamples=48):
    
    spikes=[]
        
    for i in range(0,len(electrode)-noSamples, noSamples):
        
        peak= max(electrodeAbs[i:i+noSamples])
        #print(peak)
        
        if peak>= thresh :
            #print('YES BIGGER THAN THRESH')
            peakIndex=np.where(electrodeAbs[i:i+noSamples]== peak)
            peakIndex[0][0]+=i

            #print('I AM PEAK INDEXXXXXXX ', peakIndex,peakIndex[0][0])
            #np.append(spikes,electrode[peakIndex[0][0]-noSamples//2,peakIndex[0][0]+noSamples//2])
            #print(spikes)
            #print(electrode[(29-noSamples//2):(29+noSamples//2)])
            spikes.append(electrode[(peakIndex[0][0]-noSamples//2):(peakIndex[0][0]+noSamples//2)])
        else:
            pass
    return np.array([spikes])




            
        
spikes=getSpikes (electrode1Smooth,electrode1, thresh1)
print('NUMBER OF SPIKES',spikes.shape)

#plt.plot(np.array([range(0,48)]), spikes)
i=0
for x in spikes:


    plt.plot(x)
    print(i)
    i+=1
    if i >= 3:
    	break

plt.show()
#____________________________________________________________________________________________________________

def alignSpikes (spikes, alignmentIndex=20):
	shiftedSpikes=[]

	for i in range(spikes.shape[0]):

	

		maxindex= np.argmax(spikes[i])
		

		shiftAmount= -1* (maxindex- alignmentIndex +1)
		

		shiftedSpikes.append(shift(spikes[i], shiftAmount, cval=0))
		#print(shiftedSpikes[i])

	return np.array(shiftedSpikes)



# output=alignSpikes(spikes)

# for x in output:


# 	plt.plot(x)
# 	#print(i)
# plt.show()
