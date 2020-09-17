import MDAnalysis as mda
import multiprocessing
import math as m

############################################
### A class for defining a trajectory

class trajectory:
	def __init__(self, PSF, DCD, Segnames):
		self.DCD  = DCD
		self.PSF  = PSF
		self.Seg1 = Segnames[0]
		self.Seg2 = Segnames[1]
		self.Seg3 = Segnames[2]
		self.Seg4 = Segnames[3]

		self.u = mda.Universe(self.PSF,self.DCD)

	def reading_traj(self):
		self.X = self.u.dimensions[0]
		self.Y = self.u.dimensions[1]
		return self.u, self.X, self.Y
	
	def selections(self):
		self.CA1 = self.u.select_atoms("protein and name CA and (segid " + self.Seg1 + " or segid " + self.Seg3 + " )")
		N1  = len(self.CA1)
		self.CA2 = self.u.select_atoms("protein and name CA and (segid " + self.Seg2 + " or segid " + self.Seg4 + " )")
		N2  = len(self.CA2)
		return self.CA1, N1, self.CA2, N2

	def sub_selections(self, i, j, step):

		i1 = (i * step)
		i2 = int((i + 1) * step)

		j1 = (j * step)
		j2 = int((j + 1) * step)

		sub1 = self.u.select_atoms("protein and name CA and (segid " + self.Seg1 + " or segid " + self.Seg3 + " ) and resid " + str(i1) + ":" + str(i2))
		sub2 = self.u.select_atoms("protein and name CA and (segid " + self.Seg2 + " or segid " + self.Seg4 + " ) and resid " + str(j1) + ":" + str(j2))

		return sub1, sub2


#########################################################
### A function to calcualte the distance
def distance(x1, x2, y1, y2, z1, z2, L_x_half, L_y_half, L_x, L_y):
	d_x = abs(x1 - x2)
	d_y = abs(y1 - y2)
	d_z = abs(z1 - z2)

	if d_x > L_x_half:
		d_x = L_x - d_x
	if d_y > L_y_half:
		d_y = L_y - d_y

	d = (d_x * d_x + d_y * d_y + d_z * d_z)**0.5
	return d
		