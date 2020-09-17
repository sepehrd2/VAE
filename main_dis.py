import numpy as np
import trajectory as my_traj
import multiprocessing
import time
###############################################################################################
### PSF and DCD files
directory = "/Scr/atrifan2/RAS/Kras/mri_KRAS/Kras/DIMER/dimers_1"
PSF       = directory + "/build/system.psf"
DCD       = "reduced.dcd"
step      = 5
###############################################################################################
### Reading the trajectory - size of the box - number of frames

print ("Reading the trajectory ....")
print ("-------------------------------------")
KRAS_trajectory = my_traj.trajectory(PSF , DCD, ["RAS1", "RAS3", "L1", "L3"]) 
frames          = KRAS_trajectory.reading_traj()[0]
X_length        = KRAS_trajectory.reading_traj()[1]
Y_length        = KRAS_trajectory.reading_traj()[2]
N_frames        = len(frames.trajectory)
X_length_half   = 0.5 * X_length
Y_length_half   = 0.5 * Y_length

print ("Trajectory is read :-)" )
print ("Number of frames: " + str(N_frames))
print ("The dimention of the box in x-y plane: x = " + str(X_length) + " and y = " + str(Y_length))
print ("-------------------------------------")

###############################################################################################
### Selection of CA atoms in each monomer

[CA1, N1, CA2, N2] = KRAS_trajectory.selections()

print ("Number of residues in Segid 1 = " + str(N1))
print ("Number of residues in Segid 2 = " + str(N2))
print ("-------------------------------------")

N1 = int(N1 / step)
N2 = int(N2 / step)
# N2 = 10
### An array to save residue pariwise distances

Residue_d = np.zeros((N1, N2, N_frames))
D_d = np.zeros((N_frames))

print ("Allocating Memory for storing the data")
print ("-------------------------------------")

###############################################################################################
## A Function to parallelize

def distance_per_res(argument):
	global X_length
	global Y_length
	global frames
	global D_d

	i = int(argument[0])
	j = int(argument[1])
	[CA1_sub, CA2_sub] = KRAS_trajectory.sub_selections(i, j, step)
	# OUTPUT  =  open(str(i) + "-"  + str(j) + ".txt" , "w")
	for ts in frames.trajectory:
		[x1, y1, z1] = CA1_sub.center_of_mass()

		[x2, y2, z2] = CA2_sub.center_of_mass()
		D_d[ts.frame] = my_traj.distance(x1, x2, y1, y2, z1, z2, X_length_half, Y_length_half, X_length, Y_length)
	# 	OUTPUT.write('{}    '.format(ts.frame))
	# 	OUTPUT.write('{}  \n'.format(D_d[ts.frame]))
	# OUTPUT.close()
	print (str(i) + "-" + str(j) + " out of " + str(N1) + "-" + str(N2))
	return D_d

###############################################################################################
### An array of indices containing all the CA atoms from each monomer

indecies = np.zeros((N1 * N2, 2))
for i in range(0, N1):
	for j in range(0, N2):
		indecies[i * N2 + j][0] = i
		indecies[i * N2 + j][1] = j

###############################################################################################
### Parallelization

num_cores  = 2
#multiprocessing.cpu_count()
num_chunks = int(N1 * N2 / num_cores) + 1
# OMP_NUM_THREADS=1 
a_pool = multiprocessing.Pool(num_cores)
### Deviding indecies into different chunks

start_time = time.time()
for i in range(0, num_chunks):
	k1 = int(i * num_cores)
	k2 = int((i + 1) * num_cores)
	if k2 > (N1 * N2):
		k2 = N1 * N2
	sub_indecies = []
	sub_indecies = np.zeros((k2 - k1, 2))

	for j in range(k1, k2):
		sub_indecies[j - k1][0] = indecies[j][0]
		sub_indecies[j - k1][1] = indecies[j][1]

	Temporary = []
	Temporary = (a_pool.map(distance_per_res, sub_indecies))

	for j in range(0, len(Temporary)):
		Residue_d[int(sub_indecies[j][0])][int(sub_indecies[j][1])] = Temporary[j]
end_time = time.time()
print (end_time - start_time)
###############################################################################################
### Saving the distance array 

with open('saved_distances.npy', 'wb') as f:
  	np.save(f, Residue_d)
# print (Residue_d)
