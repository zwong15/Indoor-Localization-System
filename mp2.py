import numpy as np
import numpy.linalg as la
import csv

# Reads CSV data as floats from file w/ name, returns list of those rows
def read_csv_data(file_name):
    data = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        line_count = 0
        for row in csv_reader:
            data.append(row)
    return data

# Using accelerometer data (first row of CSV), get initial orientation
# (assumes projection of phone’s local Y axis onto hor. plane is towards the global Y axis)
def get_init_orientation(acc_unnormalized):
    # Get v1 from v2, using assumption that proj. of phone’s local Y axis onto hor. plane is towards North

    v2 = acc_unnormalized / la.norm(acc_unnormalized)
    proj_loc_y_onto_v2 = ((np.dot([0, 1, 0], v2)/(la.norm(v2)**2)) * v2)

    proj_loc_y_onto_hor = [0, 1, 0] - proj_loc_y_onto_v2
    v1 = proj_loc_y_onto_hor / la.norm(proj_loc_y_onto_hor)

    # Solve R[v1 v2] = [0 0]
    #                  [1 0]
    #                  [0 1]
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

def axis_angle_to_matrix(axis, theta):
    normalized_axis = axis / la.norm(axis)
    [ux, uy, uz] = normalized_axis
    # Source: https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    R = np.array([[np.cos(theta)+((ux**2)*(1-np.cos(theta))), (ux*uy*(1-np.cos(theta)))-(uz*np.sin(theta)), (ux*uz*(1-np.cos(theta)))+(uy*np.sin(theta))],
                  [(ux*uy*(1-np.cos(theta)))+(uz*np.sin(theta)), np.cos(theta)+((uy**2)*(1-np.cos(theta))), (uy*uz*(1-np.cos(theta)))-(ux*np.sin(theta))],
                  [(ux*uz*(1-np.cos(theta)))-(uy*np.sin(theta)), (uy*uz*(1-np.cos(theta)))+(ux*np.sin(theta)), np.cos(theta)+((uz**2)*(1-np.cos(theta)))]])
    return R

def write_vector_to_file(vector, file):
    # Round entries to 4 decimal places (X.XXXX)
    new_vector = [round(x, 4) for x in vector]
    write_file.write('    '.join(['{:.4f}'.format(x) for x in new_vector]))
    write_file.write('\n')

write_file = open("result.txt", "w+")
data = read_csv_data('data.csv')


# --------- PART 1 ---------

# First row of data has initial accelerometer reading to get orientation
init_acc_data = data[0]
R0 = get_init_orientation(init_acc_data)
init_x_dir = R0[:,0]
write_vector_to_file(init_x_dir, write_file)
# print(R0)

# --------- PART 2 ---------

gyro_data = data[1:]
delta_t = 0.01
Ri = R0
for l in gyro_data:
    delta_theta = np.sqrt(l[0]**2 + l[1]**2 + l[2]**2) * delta_t
    # Project instant rotation axis l into global frame
    proj_l = Ri @ l
    delta_Ri = axis_angle_to_matrix(proj_l, delta_theta)
    # R(i+1) = delta_Ri * Ri
    Ri = delta_Ri @ Ri

final_x_dir = Ri[:,0]
write_vector_to_file(final_x_dir, write_file)
write_file.close()

print(R0)
print(Ri)
