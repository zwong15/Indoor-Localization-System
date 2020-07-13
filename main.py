import csv
import matplotlib.pyplot as plt
import statistics
import numpy as np
import numpy.linalg as la
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter,filtfilt, find_peaks, lfilter

PLOT_ROUTES = True
PLOT_ROUTES_SAME = True
PLOT_COLORS = ['blue', 'orange', 'green', 'red', 'cyan', 'magenta', 'black']
TESTING = True

# Sampling frequency
f_sampling = 100
# Nyquist frequency
f_nyquist = 0.5 * f_sampling
# Cutoff frequency we want for low-pass filter
f_cutoff_low = 2.4
f_cutoff_low_ratio = f_cutoff_low / f_nyquist
# Cutoff frequency we want for high-pass filter
f_cutoff_high = 2
f_cutoff_high_ratio = f_cutoff_high / f_nyquist
order = 2
delta_t = 0.01

peak_threshold=0.00001
peak_prominence=1.0
# Peaks could probably only happen every 0.3s (at the fastest)
peak_distance=0.3/0.01

def read_data(file_name):
    data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        for row in csv_reader:
            if not('' in row):
                data.append(row)
    return data

# Uses scipy.butter to apply low-pass filter
def low_pass_filter(data):
    # f_cutoff_ratio gives ratio of the <desired frequency = 2Hz> : <Nyquist frequency = 0.5*100Hz>
    b, a = butter(order, f_cutoff_low_ratio, btype='lowpass', analog=False)
    return filtfilt(b, a, data)
# Uses scipy.butter to apply high-pass filter
def high_pass_filter(data):
    # f_cutoff_ratio gives ratio of the <desired frequency = 2Hz> : <Nyquist frequency = 0.5*100Hz>
    b, a = butter(order, f_cutoff_high_ratio, btype='highpass', analog=False)
    return filtfilt(b, a, data)

# Get magnitude of row by using Pythagorean on x,y,z (1,2,3) IMU data
def magnitude(row):
    return ( (row[1]**2) + (row[2]**2) + (row[3]**2) )**(0.5)

# Using accelerometer data (first row of CSV), get initial orientation
# (assumes projection of phone’s local Y axis onto hor. plane is towards the global Y axis)
def get_init_orientation(acc_unnormalized):
    # Get v1 from v2, using assumption that proj. of phone’s local Y axis onto hor. plane is towards North
    v2 = acc_unnormalized / la.norm(acc_unnormalized)
    proj_loc_y_onto_v2 = ((np.dot([0, 1, 0], v2)/(la.norm(v2)**2)) * v2)
    proj_loc_y_onto_hor = [0, 1, 0] - proj_loc_y_onto_v2
    v1 = proj_loc_y_onto_hor / la.norm(proj_loc_y_onto_hor)
    [ax, ay, az] = v2
    [mx, my, mz] = v1
    a = my*az - ay*mz
    b = -(mx*az - ax*mz)
    c = mx*ay - ax*my
    # Solve system of equations: v1*row1 = 0; v2*row2 = 0; det(R) = 1
    A = np.array([v1, v2, [a, b, c]])
    row1 = la.solve(A, [0,0,1])
    R = np.array([row1, v1, v2])
    return R

# Get all gyro readings between time_1 and time_2
def get_gyro_readings_between_times(time_1, time_2):
    gyros = list(filter(lambda row: row[0] > time_1 and row[0] <= time_2, gyro_data))
    return [gyro[1:] for gyro in gyros]

# Get all delta_R_gyro readings (from large array), between time_1 and time_2
def get_gyro_delta_Rs_between_times(time_1, time_2, all_delta_Rs):
    start_index = ts_to_index(time_1)
    end_index = ts_to_index(time_2)
    return all_delta_Rs[start_index : end_index]

# Convert axis + angle to rotation matrix form
def axis_angle_to_matrix(axis, theta):
    normalized_axis = axis / la.norm(axis)
    [ux, uy, uz] = normalized_axis
    c = np.cos(theta)
    s = np.sin(theta)
    # Source: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = np.array([[c+((ux**2)*(1-c)),    (ux*uy*(1-c))-(uz*s), (ux*uz*(1-c))+(uy*s)],
                  [(ux*uy*(1-c))+(uz*s), c+((uy**2)*(1-c)),    (uy*uz*(1-c))-(ux*s)],
                  [(ux*uz*(1-c))-(uy*s), (uy*uz*(1-c))+(ux*s), c+((uz**2)*(1-c))]])
    return R

# Multiply all gyro_delta_Rs together and return
def integrate_gyro_delta_Rs(gyro_delta_Rs):
    delta_R_for_step = np.eye(3)
    for gyro_delta_R in gyro_delta_Rs:
        delta_R_for_step = gyro_delta_R @ delta_R_for_step
    return delta_R_for_step

# Integrate all gyro readings, put these matrices in a (massive) array, return
def integrate_all_gyros(gyro_data, R):
    Ri = R0
    all_Rs = []
    all_delta_Rs = []
    for l in gyro_data:
        delta_theta = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2) * delta_t
        proj_l = Ri @ l
        delta_Ri = axis_angle_to_matrix(proj_l, delta_theta)
        Ri = delta_Ri @ Ri
        all_Rs.append(Ri)
        all_delta_Rs.append(delta_Ri)
    return all_Rs, all_delta_Rs

# Convert rotation matrix to axis & angle
def matrix_to_axis_angle(A):
    a,b,c = A[0,0],A[0,1],A[0,2]
    d,e,f = A[1,0],A[1,1],A[1,2]
    g,h,i = A[2,0],A[2,1],A[2,2]
    axis = np.array([h-f, c-g, d-b])
    angle = np.arcsin(la.norm(axis) / 2)
    axis = axis / la.norm(axis)
    return (axis,angle)

# Use both angle and time to inform step length
def get_step_length(angle, time):
    angle = np.abs(angle)
    time = np.abs(time)
    time = time/1000
    return (0.762*angle) +(0.25/(time**2))

# If person walking faster than 1 step every 0.5s
# Then their stride length is at least a meter
# Step time may inversely affect length
def get_step_length_1(time):
    time = time/1000
    return (0.25/time**2)

# EAAR approach: get dh of head, use that to inform step length
# Maybe not applicable since IMU not on head
def get_step_length_2(angle):
    angle = np.abs(angle)
    delta_h = 0.8636 - ((0.8636)*np.cos(angle))
    step_length = 2 * delta_h * (np.sin(angle)/(1-np.cos(angle)))
    return step_length

def get_step_length_3(angle):
    return 0.762*np.abs(angle)

# Used to round each timestamp down to the nearest 0.01s
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# Source: https://www.kite.com/python/answers/how-to-rotate-a-3d-vector-about-an-axis-in-python
def get_perpendicular(v):
    # Rotate v around [0,0,-1] by 90 degrees
    rot_radians = np.radians(90)
    rot_axis = np.array([0, 0, 1])

    rot_vector = rot_radians * rot_axis
    rotation = R.from_rotvec(rot_vector)
    rotated_v = rotation.apply(v)
    return rotated_v

# Project vector onto horizontal plane
def project_onto_hor(v):
    grav = np.array([0,0,-1])
    proj_onto_grav = (np.dot(v, grav)/(la.norm(grav))) * grav
    proj_onto_hor = v - proj_onto_grav
    proj_onto_hor = proj_onto_hor / la.norm(proj_onto_hor)
    return proj_onto_hor

# Project all acc. data into global frame
def get_acc_data_global(acc_data, all_Rs):
    acc_data_global = []
    for i in range(len(all_Rs)):
        time = acc_data[i][0]
        current_R = all_Rs[i]
        current_acc = np.array(acc_data[i][1:])
        current_acc_g = current_R @ current_acc
        acc_data_global.append([time, current_acc_g[0], current_acc_g[1], current_acc_g[2]])
    return acc_data_global

# Convert to / from timestamp / index
def ts_to_index(timestamp):
    return int(timestamp/10)
def index_to_ts(index):
    return float(index*10)

# Find outliers in dataset using std. dev.
def find_outliers(data):
    anomalies = []
    # Set upper and lower limit to 3 standard deviation
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 3
    lower_limit  = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    # Generate outliers
    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies

# Get nth maximum value of data, removing outliers first
def get_nth_max_val(n, data):
    # Gets nth maximum of data with outliers REMOVED
    outliers = find_outliers(data)
    data_no_outliers = list(filter(lambda d: not(d in outliers), data))
    data_no_outliers.sort()
    nth_max = data_no_outliers[len(data_no_outliers) - n]
    return nth_max

# Get statistical upper limit of data using std. dev., mean
def get_upper_limit(data):
    data_std = np.std(data)
    data_mean = np.mean(data)
    anomaly_cut_off = data_std * 3
    lower_limit  = data_mean - anomaly_cut_off
    upper_limit = data_mean + anomaly_cut_off
    return upper_limit

# (Not used anymore)
# Moving average of data, attempting to smooth
def moving_average(data):
    N = 100
    cumsum, moving_aves = [0], []
    for i, x in enumerate(data, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    return moving_aves

def get_ave_separation(indices):
    total = 0
    count = 0
    for i in range(len(indices)-1):
        total += (indices[i+1] - indices[i])
        count += 1
    return total / count

def insert_new_indices(indices, ave_sep):
    for i in range(len(indices)-1):
        if (indices[i+1] - indices[i]) >= (1.7*ave_sep):
            # Insert new index between these
            new_index = (indices[i+1] + indices[i])/2
            indices = np.insert(indices, i+1, new_index)

    return indices

# Detect step indices of filtered data
def find_step_indices(filtered_data):
    # Find peaks of initial data, with no parameters
    (step_indices_temp, props) = find_peaks(filtered_data)
    peaks_temp = [filtered_data[i] for i in step_indices_temp]

    # Get 5th peak
    # We will require minimum peak height to be some fraction of this 5th peak's height
    # (Optionally) require a maximum peak height (calculated using std. dev)
        # For now, don't constrain max. height. We want to include huge peaks as they may be steps
    nth_max = get_nth_max_val(5, peaks_temp)
    min_height = 0.75*nth_max
    max_height = get_upper_limit(filtered_data)
    # Search for peaks again with specified params
    (step_indices, props) = find_peaks(filtered_data,
                                       threshold=peak_threshold,
                                       prominence=peak_prominence,
                                       distance=peak_distance,
                                       height=[min_height])

    # ave_sep = get_ave_separation(step_indices)
    # step_indices = insert_new_indices(step_indices, ave_sep)

    return step_indices

def plot_route_separate(locs, plot_index, route):
    locs = np.array(locs)
    plt.axis('equal')
    p = plt.subplot(plot_index)
    plt.plot(locs[:,0], locs[:,1])
    plt.plot(0, 0, 'gs')
    plt.title('Route ' + route)

def plot_route_same(locs, plot_index, route):
    locs = np.array(locs)
    plot_color = PLOT_COLORS[plot_index]
    plt.plot(locs[:,0], locs[:,1], color=plot_color, label=route)
    return

def plot_data_w_peaks(filtered_data, step_indices, peaks):
    plt.figure(figsize=(16,9))
    plt.plot(filtered_data, 'g')
    plt.plot(step_indices, peaks, '.')
    plt.title('Path ' + route)
    plt.show()

def write_trajectory_to_file(locs, index, route_name):
    path = ''
    if TESTING:
        path = './testing/'
    else:
        path = './training/'
    file_name = route_name + '.txt'
    total_path = path + file_name
    f = open(total_path, 'w+')
    for loc in locs:
        f.write(str(loc[0]) + ' ' + str(loc[1]) + '\n')
    f.close()

# -------- MAIN CODE --------
routes = []
if TESTING:
    routes = ['1', '2', '3', '4', '5', '6']
# TRAINING
else:
    routes = ['1', '2', '3', '4']
labels = [('Route ' + route) for route in routes]

plot_index = 0
if PLOT_ROUTES:
    plt.figure(figsize=(16,9))
    # Plotting all routes in same graph
    if PLOT_ROUTES_SAME:
        plt.axis('equal')
        plt.title('All Route Trajectories')
    # Plotting all routes side-by-side
    else:
        plot_index = 221

index = 0
for route in routes:
    acc_filename, gyro_filename = '', ''

    if TESTING:
        acc_filename = './data_imu_loc_testing/route' + route + '/Accelerometer.csv'
        gyro_filename = './data_imu_loc_testing/route' + route + '/Gyroscope.csv'
    # TRAINING
    else:
        acc_filename = './data_imu_loc/route' + route + '/Accelerometer.csv'
        gyro_filename = './data_imu_loc/route' + route + '/Gyroscope.csv'

    # ---- Read in data ----
    acc_data = read_data(acc_filename)
    gyro_data = read_data(gyro_filename)

    # Trim down data so they are the same # samples
    min_length = min(len(acc_data), len(gyro_data))
    acc_data = acc_data[:min_length]
    gyro_data = gyro_data[:min_length]

    init_acc_data = acc_data[0][1:]
    R0 = get_init_orientation(init_acc_data)
    gyro_data_no_times = [row[1:] for row in gyro_data]
    all_Rs, all_delta_Rs = integrate_all_gyros(gyro_data_no_times, R0)

    print(route)

    # Round times to nearest 0.01 s (regard as constant 100Hz)
    acc_data = [[truncate(row[0], 1), row[1], row[2], row[3]] for row in acc_data]
    gyro_data = [[truncate(row[0], 1), row[1], row[2], row[3]] for row in gyro_data]
    acc_times = [row[0] for row in acc_data]
    gyro_times = [row[0] for row in gyro_data]

    acc_data_global = get_acc_data_global(acc_data, all_Rs)
    acc_data_global_zs = [row[3] for row in acc_data_global]
    # z_avg = np.mean(acc_data_global_zs)
    # acc_data_global_zs = [(z - z_avg) for z in acc_data_global_zs]

    # ---- Filter & count steps from acc_data----
    filtered_data = low_pass_filter(acc_data_global_zs)
    step_indices = find_step_indices(filtered_data)
    step_times = [acc_times[i] for i in step_indices]
    peaks = [filtered_data[i] for i in step_indices]

    print(step_times[0])

    if not(PLOT_ROUTES):
        plot_data_w_peaks(filtered_data, step_indices, peaks)

    # ---- Track walking direction & next locations step-wise ----
    locs = []
    locs.append(np.array([0,0,0]))
    even = True

    times_between_steps = []

    for i in range(len(step_times) - 1):
        gyro_delta_Rs = get_gyro_delta_Rs_between_times(step_times[i], step_times[i+1], all_delta_Rs)
        delta_R_for_step = integrate_gyro_delta_Rs(gyro_delta_Rs)

        (axis,angle) = matrix_to_axis_angle(delta_R_for_step)
        axis = project_onto_hor(axis)
        axis = get_perpendicular(axis)

        # Flip axis on alternating steps
        if (even):
            axis = np.array([-1*axis[0], -1*axis[1], axis[2]])

        walking_dir = axis
        time_between_steps = step_times[i+1] - step_times[i]
        step_length = get_step_length_1(time_between_steps)
        disp_of_step = step_length * walking_dir
        locs.append(locs[i] + disp_of_step)

        times_between_steps.append(time_between_steps)
        even = not(even)

    print(np.mean(times_between_steps[:20])/1000)

    write_trajectory_to_file(locs, index, route)
    # Plot routes
    if PLOT_ROUTES:
        if PLOT_ROUTES_SAME:
            plot_route_same(locs, plot_index, route)
        else:
            plot_route_separate(locs, plot_index, route)
        plot_index += 1

    index += 1

if PLOT_ROUTES:
    if PLOT_ROUTES_SAME:
        plt.legend(labels)
    plt.show()
