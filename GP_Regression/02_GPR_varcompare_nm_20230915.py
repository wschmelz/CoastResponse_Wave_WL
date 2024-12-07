import os
import sys
import time
from time import strftime
import csv
import glob
import numpy
from numba import njit
import torch
import points_within

print(sys.version)

is_cuda_available = torch.cuda.is_available()
print("CUDA Available:", is_cuda_available)

def get_available_device():
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
            return torch.device('cuda:0')
        except RuntimeError:
            pass
        try:
            torch.cuda.set_device(1)
            return torch.device('cuda:1')
        except RuntimeError:
            pass
    return torch.device('cpu')

#device = get_available_device()
device = torch.device('cpu')
print(f"Using device: {device}")

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
msk_wkspc = wkspc + '01_Masks/'

print (str(__file__))
print (time.strftime("%H:%M:%S"))

@njit
def matern(sigma,numer,denom,data):
	
	if numer >= 5.0:
		ret_val = (sigma**2.) * (1.+ ((numpy.sqrt(5.) * (data))/denom) + ((5. * (data**2))/(3.*(denom**2.)))) *  numpy.exp(-1.*((numpy.sqrt(5.) * (data))/denom)) 
	if numer > 1.0 and numer <5.0:
		ret_val = (sigma**2.) * (1.+((numpy.sqrt(3.) * (data))/denom)) *  numpy.exp(-1.*((numpy.sqrt(3.) * (data))/denom)) 
	if numer <= 1.0:
		ret_val = (sigma**2.) * numpy.exp(-1.*((1. * (data))/denom))
	return ret_val 
@njit	
def linear1(sigma1,sigma2,data):
	
	ret_val = ((sigma1**2.))  + (((sigma2**2.)) * data)
	
	return ret_val 	
@njit		
def linear2(sigma1,sigma2,sigma3,sigma4,sigma5,sigma6,data,s_mat1,s_mat2,s_mat3):
	
	ret_val = ((sigma1**2.) * s_mat1) + ((sigma2**2.)* s_mat2) + ((sigma3**2.)* s_mat3) + ((((sigma4**2.) * s_mat1) + ((sigma5**2.) * s_mat2) + ((sigma6**2.) * s_mat3)) * data)
	
	return ret_val 	
@njit	
def MATERN_X_2(hyp1_in,hyp2_in,t_matrix_i,mat_deg_in,output_mat):

	ls_time_t = 1.0
		
	new_dists_t = numpy.sqrt((t_matrix_i/(ls_time_t))**2.)

	return matern(hyp1_in,mat_deg_in,numpy.absolute(hyp2_in),new_dists_t) 
@njit
def WHE_NSE(hyp1_in,hyp2_in,t_matrix_i,noise_mat_in,s_mat1,s_mat2):
	
	return (noise_mat_in) + ((s_mat1*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp1_in **2.)) +  ((s_mat2*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp2_in **2.))  #+ ((s_mat3*(numpy.identity(len(t_matrix_i[:,0])))) * (hyp3_in **2.))

file_out = wkspc + 'processed_data_LOESS.csv'	
topo_data_combined = numpy.genfromtxt(file_out,delimiter=',')

min_northing = numpy.min(topo_data_combined[:, 2])
max_northing = numpy.max(topo_data_combined[:, 2])

file_out = msk_wkspc + 'Cutoff_1103.csv'	
comp_bound2 = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound2)
w_del = numpy.where((topo_data_combined[:, 0]==17.)&(mask_flat==False))[0]

topo_data_combined = numpy.delete(topo_data_combined,w_del,axis=0)

file_out = msk_wkspc + 'Cutoff_0127.csv'	
comp_bound2 = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound2)
w_del = numpy.where((topo_data_combined[:, 0]==102.)&(mask_flat==False))[0]

topo_data_combined = numpy.delete(topo_data_combined,w_del,axis=0)

file_out = msk_wkspc + 'Cutoff_0217.csv'	
comp_bound2 = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound2)
w_del = numpy.where((topo_data_combined[:, 0]==123.)&(mask_flat==False))[0]

topo_data_combined = numpy.delete(topo_data_combined,w_del,axis=0)

file_out = msk_wkspc + 'Cutoff_0327.csv'	
comp_bound2 = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound2)
w_del = numpy.where((topo_data_combined[:, 0]==161.)&(mask_flat==False))[0]

topo_data_combined = numpy.delete(topo_data_combined,w_del,axis=0)

file_out = msk_wkspc + 'Cutoff_0406.csv'	
comp_bound1 = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound1)
w_del = numpy.where((topo_data_combined[:, 0]==191.)&(mask_flat==False))[0]

topo_data_combined = numpy.delete(topo_data_combined,w_del,axis=0)

new_points = []

current_northing = min_northing
while current_northing <= max_northing + 10.:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & (topo_data_combined[:, 2] < current_northing + 10.0))[0]

	if in_window.size > 0:
		
		percentile_97_value = numpy.percentile(topo_data_combined[in_window, 1], 99.5)

		new_easting = percentile_97_value
		zero_val = 0.04 * numpy.random.randn()
		new_point = [new_easting, current_northing + 5.0] 
		new_points.append(new_point)

	current_northing += 10.0

current_northing = max_northing
while current_northing >= min_northing - 10.:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing - 10.0) & (topo_data_combined[:, 2] < current_northing))[0]

	if in_window.size > 0:
		
		percentile_97_value = numpy.percentile(topo_data_combined[in_window, 1], 1)

		new_easting = percentile_97_value
		zero_val = 0.04 * numpy.random.randn()
		new_point = [new_easting, current_northing + 5.0]
		new_points.append(new_point)

	current_northing -= 10.0

new_points_array1 = numpy.array(new_points)

new_points = []

current_northing = min_northing
while current_northing <= max_northing:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & (topo_data_combined[:, 2] < current_northing + 50.0))[0]

	if in_window.size > 0:
		easternmost_index = in_window[numpy.argmax(topo_data_combined[in_window, 1])]

		new_easting = topo_data_combined[easternmost_index, 1] + 10.0
		zero_val = 0.025 * numpy.random.randn()
		new_point = [0, new_easting, current_northing + 25.0, zero_val, 1.0,0.0]
		new_points.append(new_point)

	current_northing += 50.0

unique_times = numpy.unique(topo_data_combined[:, 0])

new_points_array = numpy.array(new_points)

for time1 in unique_times:
    temp_points = new_points_array.copy()
    temp_points[:, 0] = time1
    topo_data_combined = numpy.append(topo_data_combined, temp_points,axis=0)

new_points = []

current_northing = min_northing
while current_northing <= max_northing:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & (topo_data_combined[:, 2] < current_northing + 50.0))[0]

	if in_window.size > 0:
		easternmost_index = in_window[numpy.argmax(topo_data_combined[in_window, 1])]

		new_easting = topo_data_combined[easternmost_index, 1] + 40.0
		zero_val = 0.025 * numpy.random.randn()
		new_point = [0, new_easting, current_northing + 25.0, zero_val, 1.0,0.0]
		new_points.append(new_point)

	current_northing += 50.0

unique_times = numpy.unique(topo_data_combined[:, 0])

new_points_array = numpy.array(new_points)

for time1 in unique_times:
    temp_points = new_points_array.copy()
    temp_points[:, 0] = time1 
    topo_data_combined = numpy.append(topo_data_combined, temp_points,axis=0)

w_high = numpy.where(numpy.abs(topo_data_combined[:,3]) > 3.0)[0]
topo_data_combined = numpy.delete(topo_data_combined,w_high,axis=0)

x_mean = numpy.mean(topo_data_combined[:,1])
y_mean = numpy.mean(topo_data_combined[:,2])
topo_data_combined[:,1] = topo_data_combined[:,1] - x_mean
topo_data_combined[:,2] = topo_data_combined[:,2] - y_mean

w_DEMs = numpy.where(topo_data_combined[:,4] == 2.0)[0]
w_profiles = numpy.where(topo_data_combined[:,4] == 0.0)[0][::3]
w_GCPs = numpy.where(topo_data_combined[:,4] == 1.0)[0]

print("GCP measurements:",len(w_GCPs))
print("Transect measurements:",len(w_profiles))
print("DEM measurements:",len(w_DEMs))

numpy.random.shuffle(w_DEMs)
numpy.random.shuffle(w_profiles)
numpy.random.shuffle(w_GCPs)

points_used = 10000
points_used_transects = 1500

index_1 = numpy.append(w_GCPs,w_profiles)
index_1 = numpy.append(index_1,w_DEMs[0:points_used])

coord1_t = topo_data_combined[index_1,0]
coord2_x = topo_data_combined[index_1,1]
coord3_y = topo_data_combined[index_1,2]
coord4_z = topo_data_combined[index_1,3]
coord5_type = topo_data_combined[index_1,4]

MCMC_iteratoins = 1

time_array_tmp = numpy.unique(topo_data_combined[:,0])

xmin = numpy.floor(numpy.min(topo_data_combined[:,1]))
xmax = numpy.ceil(numpy.max(topo_data_combined[:,1]))
ymin = numpy.floor(numpy.min(topo_data_combined[:,2]))
ymax = numpy.ceil(numpy.max(topo_data_combined[:,2]))

cellsize_x = 2.5
cellsize_y = 2.5

xi = numpy.arange(xmin, xmax+cellsize_x,cellsize_x)
yi = numpy.arange(ymin, ymax+cellsize_y,cellsize_y)

X, Y = numpy.meshgrid(xi, yi)

x_flat = numpy.ndarray.flatten(X)
y_flat = numpy.ndarray.flatten(Y)

file_out = msk_wkspc + 'Exclusion_pts2.csv'	
comp_bound = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(x_flat + x_mean, y_flat+ y_mean, new_points_array1)

x_new = x_flat[mask_flat]
y_new = y_flat[mask_flat]

location_array_tmp = numpy.zeros((len(x_new),2))
location_array_tmp[:,0] = x_new
location_array_tmp[:,1] = y_new

time_array_tmp2 = numpy.zeros(len(location_array_tmp[:,0]))

for n in range(0,len(time_array_tmp)):

	if n == 0:
		t_vals_orig = time_array_tmp2 + time_array_tmp[n]
		location_array_x_orig = location_array_tmp[:,0] * 1.0
		location_array_y_orig = location_array_tmp[:,1] * 1.0

	if n >0:
		t_vals_orig = numpy.append(t_vals_orig,time_array_tmp2 + time_array_tmp[n])
		location_array_x_orig = numpy.append(location_array_x_orig,location_array_tmp[:,0],axis=0)
		location_array_y_orig = numpy.append(location_array_y_orig,location_array_tmp[:,1],axis=0)

points_used_2 = 20000

w_pts2 = numpy.arange(0,len(t_vals_orig),1)

numpy.random.shuffle(w_pts2)

index_2 = w_pts2[0:points_used_2]

t_vals = t_vals_orig[index_2]
location_array_x = location_array_x_orig[index_2]
location_array_y = location_array_y_orig[index_2]

s_vals = (t_vals * 0.0) + 1.0

output_matrix_1 = numpy.zeros((MCMC_iteratoins,len(t_vals)))
output_matrix_2 = numpy.zeros((MCMC_iteratoins,len(t_vals)))
output_matrix_3 = numpy.zeros((MCMC_iteratoins,len(t_vals)))
output_matrix_4 = numpy.zeros((MCMC_iteratoins,len(t_vals)))

m1 =len(t_vals)

newy_mat = numpy.zeros((1,m1))
newy_p_mat = numpy.zeros((1,m1))

K_data = numpy.array([7.27911885e-04,1.05206000e-08,5.27509183e-01,5.00000012e+01
,2.30143081e+01,1.91819558e+02,1.00000000e+00,1.69106269e-01
,2.39318380e+01,9.13065306e+01,5.93583736e-02,1.22259779e-03
,1.28907474e-01,6.78932723e+00])

hyperparams = numpy.reshape(K_data,(1,len(K_data)))
K_data = numpy.reshape(K_data,(1,len(K_data)))

w1 = numpy.where(numpy.isnan(hyperparams[:,0]))[0]

K_data = numpy.delete(K_data,w1,axis=0)

n_SL_1 = len(coord1_t)

y_1 = numpy.reshape(coord4_z,(-1,1))	
y_transpose_1 = numpy.transpose(y_1)

noise_mat_1 = (1.*(numpy.identity(n_SL_1)))

for index1 in range(0,n_SL_1):
	noise_mat_1[index1,index1] = noise_mat_1[index1,index1] * 0.0

###

t_matrix_1 = numpy.repeat(numpy.reshape(coord1_t,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(coord1_t,(-1,1)),n_SL_1,axis=1)	

###

s_matrix_1_tmp = numpy.repeat(numpy.reshape(coord5_type,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(coord5_type,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp==0.)&(s_matrix_2_tmp==0.))

s_matrix_1 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1[w1[0],w1[1]] = s_matrix_1[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp==1.)&(s_matrix_2_tmp==1.))

s_matrix_2 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2[w1[0],w1[1]] = s_matrix_2[w1[0],w1[1]] + 1.0

noise_mat_1 = noise_mat_1 + ((s_matrix_1*(numpy.identity(len(t_matrix_1[:,0])))) * (0.025 **2.)) + ((s_matrix_2*(numpy.identity(len(t_matrix_1[:,0])))) * (0.1 **2.))
###

t_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(coord1_t),axis=0) - numpy.repeat(numpy.reshape(coord1_t,(-1,1)),len(t_vals),axis=1))**2.)
t_mults_new_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(coord1_t),axis=0) * numpy.repeat(numpy.reshape(coord1_t,(-1,1)),len(t_vals),axis=1)

###

t_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1))**2.)
t_matrix2_mult_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) * numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1)

###

file_out_1 = wkspc + 'MC_probability_output_1.csv'	
file_out_2 = wkspc + 'MC_probability_output_2.csv'	
file_out_3 = wkspc + 'MC_probability_output_3.csv'	
file_out_4 = wkspc + 'MC_probability_output_4.csv'	

int_list = numpy.arange(0,len(K_data[:,0]),1)
numpy.random.shuffle(int_list)

print (len(t_vals))

t1 = float(time.time())

for MCMC_index in range(0,MCMC_iteratoins):
	numpy.random.shuffle(w_pts2)

	index_2 = w_pts2[0:points_used_2]

	t_vals = t_vals_orig[index_2]
	location_array_x = location_array_x_orig[index_2]
	location_array_y = location_array_y_orig[index_2]

	hyp1,hyp2,hyp7,hyp8,hyp9,hyp10,hyp11,hyp12,hyp13,hyp14,hyp15,hyp16,hyp17,hyp18 =  K_data[int_list[MCMC_index],:]
	
	numpy.random.shuffle(w_DEMs)
	numpy.random.shuffle(w_profiles)
	numpy.random.shuffle(w_GCPs)

	points_used = 10000
	points_used_transects = 1500

	index_1 = numpy.append(w_GCPs,w_profiles)
	index_1 = numpy.append(index_1,w_DEMs[0:points_used])

	coord1_t = topo_data_combined[index_1,0]
	coord2_x = topo_data_combined[index_1,1]
	coord3_y = topo_data_combined[index_1,2]
	coord4_z = topo_data_combined[index_1,3]
	coord5_type = topo_data_combined[index_1,4]

	n_SL_1 = len(coord1_t)

	y_1 = numpy.reshape(coord4_z,(-1,1))	
	y_transpose_1 = numpy.transpose(y_1)

	coord2_x_tmp = (coord2_x * numpy.cos(hyp18)) - (coord3_y * numpy.sin(hyp18))
	coord3_y_tmp = (coord2_x * numpy.sin(hyp18)) + (coord3_y * numpy.cos(hyp18))
	
	location_array_x_tmp = (location_array_x * numpy.cos(hyp18)) - (location_array_y * numpy.sin(hyp18))
	location_array_y_tmp = (location_array_x * numpy.sin(hyp18)) + (location_array_y * numpy.cos(hyp18))

	###

	t_matrix_1 = numpy.repeat(numpy.reshape(coord1_t,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(coord1_t,(-1,1)),n_SL_1,axis=1)	
	x_matrix_1 = numpy.repeat(numpy.reshape(coord2_x_tmp,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(coord2_x_tmp,(-1,1)),n_SL_1,axis=1)	
	y_matrix_1 = numpy.repeat(numpy.reshape(coord3_y_tmp,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(coord3_y_tmp,(-1,1)),n_SL_1,axis=1)	

	new_mults_t_1 = numpy.repeat(numpy.reshape(coord1_t,(1,-1)),n_SL_1,axis=0) * numpy.repeat(numpy.reshape(coord1_t,(-1,1)),n_SL_1,axis=1)

	###

	t_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(coord1_t),axis=0) - numpy.repeat(numpy.reshape(coord1_t,(-1,1)),len(t_vals),axis=1))**2.)
	x_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(location_array_x_tmp,(1,-1)),len(coord2_x_tmp),axis=0) - numpy.repeat(numpy.reshape(coord2_x_tmp,(-1,1)),len(location_array_x_tmp),axis=1))**2.)
	y_new_1 = numpy.sqrt((numpy.repeat(numpy.reshape(location_array_y_tmp,(1,-1)),len(coord3_y_tmp),axis=0) - numpy.repeat(numpy.reshape(coord3_y_tmp,(-1,1)),len(location_array_y_tmp),axis=1))**2.) 
	
	t_mults_new_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),len(coord1_t),axis=0) * numpy.repeat(numpy.reshape(coord1_t,(-1,1)),len(t_vals),axis=1)

	###

	t_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1))**2.)
	x_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(location_array_x_tmp,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(location_array_x_tmp,(-1,1)),m1,axis=1))**2.)
	y_matrix2_1 = numpy.sqrt((numpy.repeat(numpy.reshape(location_array_y_tmp,(1,-1)),m1,axis=0) - numpy.repeat(numpy.reshape(location_array_y_tmp,(-1,1)),m1,axis=1))**2.)
		
	t_matrix2_mult_1 = numpy.repeat(numpy.reshape(t_vals,(1,-1)),m1,axis=0) * numpy.repeat(numpy.reshape(t_vals,(-1,1)),m1,axis=1)

	###
		
	################################
	
	K1_1 =  linear1(hyp1,hyp2,new_mults_t_1)

	new_dists_t_1 = numpy.sqrt((t_matrix_1/(numpy.absolute(hyp8)))**2.)
	new_dists_d_2 = numpy.sqrt(((x_matrix_1/numpy.absolute(hyp9))**2.) + ((y_matrix_1/numpy.absolute(hyp10))**2.))	
	new_dists_d_t_2 = numpy.sqrt(((new_dists_t_1)**2.) + ((new_dists_d_2)**2.))
	K3_1 =  matern(hyp7,3.,hyp11,new_dists_d_t_2)

	del new_dists_t_1 	
	del new_dists_d_2 	
	del new_dists_d_t_2 	
							
	new_dists_t_2 = numpy.sqrt((t_matrix_1/(numpy.absolute(hyp13)))**2.)
	new_dists_d_3 = numpy.sqrt(((x_matrix_1)**2.) + ((y_matrix_1)**2.))/numpy.absolute(hyp14)	
	new_dists_d_t_3 = numpy.sqrt(((new_dists_t_2)**2.) + ((new_dists_d_3)**2.))
	K4_1 =  matern(hyp12,1.,hyp15,new_dists_d_t_3)	

	del new_dists_t_2 	
	del new_dists_d_3 	
	del new_dists_d_t_3 
		
	WN = WHE_NSE(hyp16,hyp17,t_matrix_1,noise_mat_1,s_matrix_1,s_matrix_2)
	
	K = K1_1 + K3_1 + K4_1 + WN
		
	K_inv = numpy.linalg.inv(K)
	
	#####
	
	K1_2 =  linear1(hyp1,hyp2,t_mults_new_1)
	
	new_dists_t_1 = numpy.sqrt((t_new_1/(numpy.absolute(hyp8)))**2.)
	new_dists_d_2 = numpy.sqrt(((x_new_1/numpy.absolute(hyp9))**2.) + ((y_new_1/numpy.absolute(hyp10))**2.))	
	new_dists_d_t_2 = numpy.sqrt(((new_dists_t_1)**2.) + ((new_dists_d_2)**2.))
	K3_2 =  matern(hyp7,3.,hyp11,new_dists_d_t_2)

	del new_dists_t_1 	
	del new_dists_d_2 	
	del new_dists_d_t_2 	
				
	new_dists_t_2 = numpy.sqrt((t_new_1/(numpy.absolute(hyp13)))**2.)
	new_dists_d_3 = numpy.sqrt(((x_new_1)**2.) + ((y_new_1)**2.))/numpy.absolute(hyp14)	
	
	del x_new_1 	
	del y_new_1 		
	
	new_dists_d_t_3 = numpy.sqrt(((new_dists_t_2)**2.) + ((new_dists_d_3)**2.))
	K4_2 =  matern(hyp12,1.,hyp15,new_dists_d_t_3)	

	del new_dists_t_2 	
	del new_dists_d_3 	
	del new_dists_d_t_3 
	
	K2_f = K1_2 + K3_2 + K4_2

	new_y_p = numpy.matmul(K_inv,K2_f) 
	new_y_p_2 = numpy.matmul(numpy.transpose(K2_f),new_y_p)

	#####
	
	K1_3 =  linear1(hyp1,hyp2,t_matrix2_mult_1)
	
	new_dists_t_1 = numpy.sqrt((t_matrix2_1/(numpy.absolute(hyp8)))**2.)
	new_dists_d_2 = numpy.sqrt(((x_matrix2_1/numpy.absolute(hyp9))**2.) + ((y_matrix2_1/numpy.absolute(hyp10))**2.))	
	new_dists_d_t_2 = numpy.sqrt(((new_dists_t_1)**2.) + ((new_dists_d_2)**2.))
	K3_3 =  matern(hyp7,3.,hyp11,new_dists_d_t_2)

	del new_dists_t_1 	
	del new_dists_d_2 	
	del new_dists_d_t_2 	
	
	new_dists_t_2 = numpy.sqrt((t_matrix2_1/(numpy.absolute(hyp13)))**2.)
	new_dists_d_3 = numpy.sqrt(((x_matrix2_1)**2.) + ((y_matrix2_1)**2.))/numpy.absolute(hyp14)	
	new_dists_d_t_3 = numpy.sqrt(((new_dists_t_2)**2.) + ((new_dists_d_3)**2.))
	K4_3 =  matern(hyp12,1.,hyp15,new_dists_d_t_3)	
	
	del new_dists_t_2 	
	del new_dists_d_3 	
	del new_dists_d_t_3 
	
	K_2 = K1_3 + K3_3 + K4_3

	new_y = numpy.matmul(numpy.transpose(K2_f),numpy.matmul(K_inv,numpy.reshape(y_1,(-1,1))))

	newy_mat[0,:] = numpy.ndarray.flatten(new_y)
	newy_p_mat[0,:] = numpy.sqrt(numpy.diag(K_2 - new_y_p_2))
	
	output_matrix_1[MCMC_index,:] = newy_mat[0,:] #+ (numpy.random.standard_normal(size=len(newy_p_mat[0,:])) * newy_p_mat[0,:])
	output_matrix_2[MCMC_index,:] = t_vals * 1.0
	output_matrix_3[MCMC_index,:] = location_array_x * 1.0
	output_matrix_4[MCMC_index,:] = location_array_y * 1.0

	if MCMC_index % 10 == 0:
	
		t2 = float(time.time())
		time_per = (t2 - t1) / (float(MCMC_index) + 1.)
		
		sys.stdout.write("\r i: %s   " % (str(numpy.round(time_per,5)))) 
		
		print ("...")

		with open(file_out_1, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_1)	
			
		with open(file_out_2, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_2)
		
		with open(file_out_3, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_3)
		
		with open(file_out_4, 'w',newline="") as csvfile2:
			writer = csv.writer(csvfile2)
			writer.writerows(output_matrix_4)			
		
print (time.strftime("%H:%M:%S"))


with open(file_out_1, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_1)	
	
with open(file_out_2, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_2)

with open(file_out_3, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_3)

with open(file_out_4, 'w',newline="") as csvfile2:
	writer = csv.writer(csvfile2)
	writer.writerows(output_matrix_4)	