# Neuron-Spike-Sorting

For Biomedical Engineering Course - Assignment 2

This assignment is to process on the data extracted from two electrodes that were implemented in a rat's motor unit.
the data represents the neuron activity (Action potentials- Spikes) of some unknown number of neurons which is obsereved by the electrodes.
the goal is to apply some sorting/clustering algorithm in order to find how many clusters we have and which neurons
belong to which cluster.

Implement the spike sorting algorithm explained in Lecture 10.pdf. Your function should take as inputs the
raw extracellular activity of multiple electrodes. The function should return a vector that contains the
timestamps of the peaks of the detected spikes for each neuron and a vector for the mean spike of each
neuron. Apply your function to the data provided at https://ufile.io/kc1yd

Each column in the data file corresponds to one electrode. The sampling rate of this data is 24414 Hz. To
detect spikes, compute the threshold as either 3.5 times the standard deviation of the first 500 samples of
each electrode, or 5 times the standard deviation of the first 500 samples of each electrode. Spikes should
be aligned based on their peak value. Extracted spikes should be of duration 2 msec, where the peak is at
the center of the extracted spike window.
