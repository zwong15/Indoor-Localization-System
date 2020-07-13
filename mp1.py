import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy.signal import butter,filtfilt, find_peaks

# Sampling frequency
f_sampling = 100
# Nyquist frequency
f_nyquist = 0.5 * f_sampling
# Cutoff frequency we want for low-pass filter
f_cutoff = 2
f_cutoff_ratio = f_cutoff / f_nyquist
order = 2

# Uses scipy.butter to apply low-pass filter
def low_pass_filter(data):
    # f_cutoff_ratio gives ratio of the <desired frequency = 2Hz> : <Nyquist frequency = 0.5*100Hz>
    b, a = butter(order, f_cutoff_ratio, btype='low', analog=False)
    return filtfilt(b, a, data)

def plot_data(xs):
    plt.plot(range(len(xs)), xs)
    plt.show()

# NOT USED: gets lower outliers of dataset using IQR
def find_outliers(xs):
    q1, q3 = np.percentile(xs,[25,75])
    iqr = q3 - q1
    #print("IQR:", iqr)
    acceptable_range = q1 - (1.5*iqr)
    #print("Acceptable range:", acceptable_range)
    outliers = []
    for x in xs:
        if x < acceptable_range:
            outliers.append(x)
    return outliers

# Get magnitude of row by using Pythagorean on all 3 dimensions of IMU data
def get_magnitude(row):
    return ( (float(row['x'])**2) + (float(row['y'])**2) + (float(row['z'])**2) )**(0.5)


# Lists to hold each col of data -> cols correspond to axes
data = []

total_filename = "data.csv"
# Start by reading in specific file
# TODO: CHANGE THIS TO 'data.csv' for their testing purposes
with open(total_filename, newline='') as csvfile:
    # Use DictReader to name the columns 'x', 'y', and 'z'
    reader = csv.DictReader(csvfile, ['x', 'y', 'z'])
    for row in reader:
        data.append(get_magnitude(row))

# Now we have the data read in
# print(data)

# Apply low-pass filter to data
# LPF has cutoff frequency = 2 Hz
filtered_data = low_pass_filter(data)

# threshold gives allowed vertical distance between a peak and its surrounding samples
(peaks_indices, props) = find_peaks(filtered_data, threshold=0.0001)

peaks = [filtered_data[i] for i in peaks_indices]
num_steps = len(peaks)
# print("# of steps:", num_steps)

# Output # of steps to 'result.txt'
f = open("result.txt", "w+")
f.write(str(num_steps))
f.close()
