#First let's get the imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans  

#Function to read two columns from file
def Read_Two_Column_File(file_name):
    with open(file_name, 'r') as data:
        x = []
        y = []
        for line in data:
            p = line.split()
            x.append(float(p[0]))
            y.append(float(p[1]))

    return x, y


#Function to smooth signals *well i eventually didn't use it*
def smoothing(squares,window_size=48):
    
    N=window_size
    smooth=np.zeros((squares.shape[0]))
    for i in range(N//2,(squares.shape[0]-(N//2))):
        
        sum=0
        for j in range (i-(N//2), i+(N//2)):
            
            sum=sum+squares[j]
        smooth[i]=(1/N)*sum
    return smooth



#Function to get the peaks, the peaks' indicies and the whole spikes
def getSpikes(electrodeAbs , electrode,thresh , noSamples=48):
    
    spikes=[]
    peaksss=[]
    peaksssIndex=[]
        
    for i in range(0,len(electrode)-noSamples, noSamples):
        
        peak= max(electrodeAbs[i:i+noSamples])
        #print(peak)
        
        if peak >= thresh :
            peaksss.append(peak)
            #print('YES BIGGER THAN THRESH')
            peakIndex=np.where(electrodeAbs[i:i+noSamples]== peak)
            peakIndex[0][0]+=i
            peaksssIndex.append(peakIndex[0][0])

            #print('I AM PEAK INDEXXXXXXX ', peakIndex,peakIndex[0][0])
            #np.append(spikes,electrode[peakIndex[0][0]-noSamples//2,peakIndex[0][0]+noSamples//2])
            #print(spikes)
            #print(electrode[(29-noSamples//2):(29+noSamples//2)])
            spikes.append(electrode[(peakIndex[0][0]-noSamples//2):(peakIndex[0][0]+noSamples//2)])
        else:
            pass
    return np.array([spikes]), np.array([peaksss]), np.array([peaksssIndex])







#THIS IS THE MAIN REQUIRED FUNCTION i.e. the pipeline of the assignment
def spikeSort(fileName , samplingRate=24414, noElectrode=1 ,sdMultiple=3.5, noClusters=2):
    
    electrode1,electrode2 = Read_Two_Column_File(fileName)
    if noElectrode==2:
        electrode=electrode2
    else:
        electrode=electrode1
        
    x_values = range(0,len(electrode))
    
    electrodeAbs= np.absolute(electrode)
    
    f1, (ax1,ax2)=plt.subplots(1,2,figsize=(24,9))
    f1.tight_layout()

    ax1.plot(x_values,electrode)
    text='Electrode'+str(noElectrode)+ 'Raw'
    ax1.set_title(text,fontsize=30)
    
    text='Electrode'+str(noElectrode)+'Absoulte'
    ax2.plot(x_values,electrodeAbs)
    ax2.set_title(text,fontsize=30)
    
    
    
    
    sd = np.std(electrode[:500])
    
    thresh= sd *sdMultiple
    
    spikes,peaksss, peaksssIndex=getSpikes (electrodeAbs,electrode, thresh)
    
    
    
    pca_elec = PCA(n_components=2)
    electrode_decomposed_spike_values= pca_elec.fit_transform(spikes[0])
    
    
    
    kmeans = KMeans(n_clusters=noClusters)   #<------------------------------  
    kmeans.fit(electrode_decomposed_spike_values)
    

    f2, (ax3,ax4)=plt.subplots(1,2,figsize=(24,9))
    f2.tight_layout()

    for x in spikes:
        
        ax3.plot(x)
    ax3.set_title('Aligned Spikes',fontsize=30)

    ax4.set_title('Feature Space',fontsize=30)
    
    ax4.scatter(electrode_decomposed_spike_values[:,0],electrode_decomposed_spike_values[:,1], c=kmeans.labels_, cmap='rainbow')  
    ax4.axis((-0.00015, 0.00025, -0.00015, 0.00025))
    
    
    
    #_______________________
    kmeans = KMeans(n_clusters=noClusters)  #<------------------------------
    spikes,peaksss, peaksssIndex=getSpikes (electrodeAbs[:20000],electrode[:20000], thresh)
    listaa=kmeans.fit_predict(spikes[0])
#     sum0=[]
#     counter0=0
#     sum1=0
#     counter1=0
#     sum2=0
#     counter2=0
#     sum3=0
#     counter3=0
    timeStamp0=[]
    timeStamp1=[]
    timeStamp2=[]
    timeStamp3=[]
    mean0=[]
    mean1=[]
    mean2=[]
    mean3=[]
    f3, (ax5)=plt.subplots(1,figsize=(24,9))
    f1.tight_layout()

    ax5.plot(electrode[:20000])
    text='Electrode'+str(noElectrode)
    ax5.set_title(text,fontsize=30)
    
#     text='Electrode'+str(noElectrode)+'Absoulte'
#     ax2.plot(x_values,electrodeAbs)
#     ax2.set_title(text,fontsize=30)
    
    for i in range(0,spikes.shape[1]):
        if listaa[i]==0:
            #print(listaa[i])

            if electrode[peaksssIndex[0][i]] != peaksss[0][i]:


                ax5.plot(peaksssIndex[0][i],(-1*peaksss[0][i]), '*',color='green' ,markerfacecolor='green')
            else:
                ax5.plot(peaksssIndex[0][i],(peaksss[0][i]), '*',color='green' ,markerfacecolor='green')
            mean0.append(spikes[0][i])

            timeStamp0.append(peaksssIndex[0][i])
            #print(peaksss[0][i])
        elif listaa[i]==1:
            #print(listaa[i])
            if electrode[peaksssIndex[0][i]] != peaksss[0][i]:
                ax5.plot(peaksssIndex[0][i],(-1*peaksss[0][i]), '*', color='blue',markerfacecolor='blue')
            else: 
                ax5.plot(peaksssIndex[0][i],peaksss[0][i], '*', color='blue',markerfacecolor='blue')
                
            mean1.append(spikes[0][i])

            timeStamp1.append(peaksssIndex[0][i])

        elif listaa[i]==2:
            #print(listaa[i])
            if electrode[peaksssIndex[0][i]] != peaksss[0][i]:
                ax5.plot(peaksssIndex[0][i],(-1*peaksss[0][i]), '*',color='red' ,markerfacecolor='red')
            else:
                ax5.plot(peaksssIndex[0][i],(peaksss[0][i]), '*',color='red' ,markerfacecolor='red')

                
            mean2.append(spikes[0][i])

            timeStamp2.append(peaksssIndex[0][i])

        elif listaa[i]==3:
            #print(listaa[i])
            if electrode[peaksssIndex[0][i]] != peaksss[0][i]:
                ax5.plot(peaksssIndex[0][i],(-1*peaksss[0][i]), '*',color='yellow',markerfacecolor= 'yellow')
            else:
                ax5.plot(peaksssIndex[0][i],peaksss[0][i], '*',color='yellow',markerfacecolor= 'yellow')
                
            mean3.append(spikes[0][i])
            timeStamp3.append(peaksssIndex[0][i])

#     print('mean of spike of class 0' , sum0/counter0)
#     print('mean of spike of class 1' , sum1/counter1)
#     print('mean of spike of class 2' , sum2/counter2)
#     print('mean of spike of class 3' , sum3/counter3)
    



    mean00= np.mean(mean0, axis=0)
    mean11= np.mean(mean1, axis=0)
    mean22= np.mean(mean2, axis=0)
    mean33= np.mean(mean3, axis=0)


#     f4, (ax7,ax8)=plt.subplots(1,2,figsize=(24,9))
#     f4.tight_layout()

#     ax7.plot(mean00)
#     text='neuron0'
#     ax7.set_title(text,fontsize=30)

#     text='neuron1'
#     ax8.plot(mean11)
#     ax8.set_title(text,fontsize=30)


#     f5, (ax9,ax10)=plt.subplots(1,2,figsize=(24,9))
#     f5.tight_layout()

#     ax9.plot(mean22)
#     text='neuron2'
#     ax9.set_title(text,fontsize=30)

#     text='neuron3'
#     ax10.plot(mean33)
#     ax10.set_title(text,fontsize=30)

    f4, (ax7)=plt.subplots(1,figsize=(24,9))
    f4.tight_layout()

    ax7.plot(mean00)
    ax7.plot(mean11)
    ax7.plot(mean22)
    ax7.plot(mean33)
    text='Template'
    ax7.set_title(text,fontsize=30)

    plt.show()










    
    
    
    return np.array( [timeStamp0,timeStamp1,timeStamp2,timeStamp3]), np.array([mean00, mean11,mean22,mean33])



timeStamp, Mean=spikeSort('Data.txt',noElectrode=2 ,sdMultiple=5, noClusters=2)
