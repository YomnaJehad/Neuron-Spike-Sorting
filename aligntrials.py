import numpy as np 
from scipy.ndimage.interpolation import shift
import matplotlib.pyplot as plt
##Trying to align the spikes
shiftedSpikes=[]
spikes =np.array([[1,2,3] ,[3,2,1] ,[4,5,1]])

for i in range(spikes.shape[0]):

	#print (spikes[i])

	maxindex= np.argmax(spikes[i])
	#print(maxindex)

	shiftAmount= -1* (maxindex- 2 +1)
	#print(shiftAmount)

	shiftedSpikes.append(shift(spikes[i], shiftAmount, cval=0))
	#print(shiftedSpikes[i])

def alignSpikes (spikes, alignmentIndex=20):
	shiftedSpikes=[]

	for i in range(spikes.shape[0]):

	

		maxindex= np.argmax(spikes[i])
		

		shiftAmount= -1* (maxindex- alignmentIndex +1)
		

		shiftedSpikes.append(shift(spikes[i], shiftAmount, cval=0))
		#print(shiftedSpikes[i])

	return np.array(shiftedSpikes)

output=alignSpikes(spikes,2)
print(output)
plt.plot([0,1,2], output)
plt.show()