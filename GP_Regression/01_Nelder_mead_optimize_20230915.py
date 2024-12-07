import os
import sys
import time
import numpy
import glob
import points_within
from numba import njit
import torch
import scipy
from scipy.optimize import minimize

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

device = torch.device('cpu')
print(f"Using device: {device}")

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))

print (time.strftime("%H:%M:%S"))

wkspc_data = wkspc + '00_Data/'
msk_wkspc = wkspc + '01_Masks/'

gpsdata = glob.glob(wkspc_data + "*/")

print ("")

@njit
def compute_mask(X, Y, topo_data_combined_orig):
	mask = numpy.ones(numpy.shape(X))
	count = 0
	count2 = 0

	for n1 in range(0, X.shape[0]):
		for n2 in range(0, X.shape[1]):
			count2 += 1
			if numpy.min(numpy.sqrt((X[n1, n2] - topo_data_combined_orig[::10, 1]) ** 2 + (Y[n1, n2] - topo_data_combined_orig[::10, 2]) ** 2)) > 7.5:
				mask[n1, n2] = numpy.nan
				count2 -= 1
			count += 1
			if (count % 10000) == 0:
				print("total pts:", count2, "; completed", count, "of", X.shape[0] * X.shape[1])
				
	return mask

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
while current_northing <= max_northing:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & 
						 (topo_data_combined[:, 2] < current_northing + 50.0))[0]

	if in_window.size > 0:
		easternmost_index = in_window[numpy.argmax(topo_data_combined[in_window, 1])]

		new_easting = topo_data_combined[easternmost_index, 1] + 10.0
		zero_val = 0.025 * numpy.random.randn()
		new_point = [0, new_easting, current_northing + 25.0, zero_val, 1.0,0.0]  # Setting time value temporarily to 0
		new_points.append(new_point)

	current_northing += 50.0

unique_times = numpy.unique(topo_data_combined[:, 0])

new_points_array = numpy.array(new_points)

for time in unique_times:
    temp_points = new_points_array.copy()
    temp_points[:, 0] = time
    topo_data_combined = numpy.append(topo_data_combined, temp_points,axis=0)

new_points = []

current_northing = min_northing
while current_northing <= max_northing:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & 
						 (topo_data_combined[:, 2] < current_northing + 50.0))[0]

	if in_window.size > 0:
		easternmost_index = in_window[numpy.argmax(topo_data_combined[in_window, 1])]

		new_easting = topo_data_combined[easternmost_index, 1] + 40.0
		zero_val = 0.025 * numpy.random.randn()
		new_point = [0, new_easting, current_northing + 25.0, zero_val, 1.0,0.0]
		new_points.append(new_point)

	current_northing += 50.0

unique_times = numpy.unique(topo_data_combined[:, 0])

new_points_array = numpy.array(new_points)

for time in unique_times:
    temp_points = new_points_array.copy()
    temp_points[:, 0] = time
    topo_data_combined = numpy.append(topo_data_combined, temp_points,axis=0)

topo_data_combined[:,1] = topo_data_combined[:,1] - numpy.mean(topo_data_combined[:,1])
topo_data_combined[:,2] = topo_data_combined[:,2] - numpy.mean(topo_data_combined[:,2])

w_high = numpy.where(numpy.abs(topo_data_combined[:,3]) > 3.5)[0]
topo_data_combined = numpy.delete(topo_data_combined,w_high,axis=0)

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

n_SL_1 = len(coord1_t)

y_1 = numpy.reshape(coord4_z,(-1,1))	
y_transpose_1 = numpy.transpose(y_1)

y_1 = torch.tensor(y_1).to(device)
y_transpose_1 = torch.tensor(y_transpose_1).to(device)

noise_mat_1 = (1.*(numpy.identity(n_SL_1)))

for index1 in range(0,n_SL_1):
	noise_mat_1[index1,index1] = noise_mat_1[index1,index1] * 0.0
	
t_matrix_1 = numpy.repeat(numpy.reshape(coord1_t,(1,-1)),n_SL_1,axis=0) - numpy.repeat(numpy.reshape(coord1_t,(-1,1)),n_SL_1,axis=1)	

s_matrix_1_tmp = numpy.repeat(numpy.reshape(coord5_type,(1,-1)),n_SL_1,axis=0)
s_matrix_2_tmp = numpy.repeat(numpy.reshape(coord5_type,(-1,1)),n_SL_1,axis=1)

w1 = numpy.where((s_matrix_1_tmp<2.)&(s_matrix_2_tmp<2.))

s_matrix_1 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_1[w1[0],w1[1]] = s_matrix_1[w1[0],w1[1]] + 1.0

w1 = numpy.where((s_matrix_1_tmp==2.)&(s_matrix_2_tmp==2.))

s_matrix_2 = numpy.zeros(numpy.shape(s_matrix_1_tmp))
s_matrix_2[w1[0],w1[1]] = s_matrix_2[w1[0],w1[1]] + 1.0

new_mults_t_1 = numpy.repeat(numpy.reshape(coord1_t,(1,-1)),n_SL_1,axis=0) * numpy.repeat(numpy.reshape(coord1_t,(-1,1)),n_SL_1,axis=1)

noise_mat_1 = noise_mat_1 + ((s_matrix_1*(numpy.identity(len(t_matrix_1[:,0])))) * (0.025 **2.)) + ((s_matrix_2*(numpy.identity(len(t_matrix_1[:,0])))) * (0.1 **2.))

@njit
def custom_repeat_2d(input_array, repeats, axis):
    if axis == 0:
        result = numpy.empty((input_array.shape[0] * repeats, input_array.shape[1]), dtype=input_array.dtype)
        for i in range(0,input_array.shape[0]):
            for j in range(0,repeats):
                result[i * repeats + j, :] = input_array[i, :]
        return result
    elif axis == 1:
        result = numpy.empty((input_array.shape[0], input_array.shape[1] * repeats), dtype=input_array.dtype)
        for i in range(0,input_array.shape[1]):
            for j in range(0,repeats):
                result[:, i * repeats + j] = input_array[:, i]
        return result
    else:
        raise ValueError("Invalid axis value.")

@njit
def compute_matrices(guess1, coord2_x, coord3_y, new_mults_t_1, t_matrix_1,s_matrix_1,s_matrix_2, noise_mat_1):

	coord2_x_tmp = (coord2_x * numpy.cos(guess1[13])) - (coord3_y * numpy.sin(guess1[13]))
	coord3_y_tmp = (coord2_x * numpy.sin(guess1[13])) + (coord3_y * numpy.cos(guess1[13]))
	
	x_matrix_1 = custom_repeat_2d(numpy.reshape(coord2_x_tmp,(1,-1)),n_SL_1,0) - custom_repeat_2d(numpy.reshape(coord2_x_tmp,(-1,1)),n_SL_1,1)
	y_matrix_1 = custom_repeat_2d(numpy.reshape(coord3_y_tmp,(1,-1)),n_SL_1,0) - custom_repeat_2d(numpy.reshape(coord3_y_tmp,(-1,1)),n_SL_1,1)
	
	K1 =  linear1(guess1[0],guess1[1],new_mults_t_1)

	new_dists_t_1 = numpy.sqrt((t_matrix_1/(numpy.absolute(guess1[3])))**2.)
	new_dists_d_2 = numpy.sqrt(((x_matrix_1/numpy.absolute(guess1[4]))**2.) + ((y_matrix_1/numpy.absolute(guess1[5]))**2.))	
	new_dists_d_t_2 = numpy.sqrt(((new_dists_t_1)**2.) + ((new_dists_d_2)**2.))
	K3 =  matern(guess1[2],3.,guess1[6],new_dists_d_t_2)
			
	new_dists_t_2 = numpy.sqrt((t_matrix_1/(numpy.absolute(guess1[8])))**2.)
	new_dists_d_3 = numpy.sqrt(((x_matrix_1)**2.) + ((y_matrix_1)**2.))/numpy.absolute(guess1[9])	
	new_dists_d_t_3 = numpy.sqrt(((new_dists_t_2)**2.) + ((new_dists_d_3)**2.))
	K4 =  matern(guess1[7],1.,guess1[10],new_dists_d_t_3)	

	WN = WHE_NSE(guess1[11],guess1[12],t_matrix_1,noise_mat_1,s_matrix_1,s_matrix_2)

	K_1 = K1 + K3 + K4 + WN

	return K_1
	
count = 0
def optimize_MLE_merge_guess(guess1):
	global count
	K_1 = compute_matrices(guess1, coord2_x, coord3_y, new_mults_t_1, t_matrix_1,s_matrix_1,s_matrix_2, noise_mat_1)
	K_1 = torch.tensor(K_1).to(device)

	t1_tmp = torch.linalg.solve(K_1, y_1)
	term1 = (-1./2.) * torch.matmul(y_transpose_1, t1_tmp)
	logdet_k = torch.slogdet(K_1)[1]
	term2 = (1./2.) * logdet_k
	term3 = (len(y_1.cpu().numpy())/2.) * numpy.log(2.*numpy.pi)
	
	opt_outs_CO2_1 = term1 - term2 - term3
	
	opt_outs_CO2 = opt_outs_CO2_1

	if opt_outs_CO2 > 0:
		opt_outs_CO2 = numpy.nan
	
	opt_outs_CO2 = opt_outs_CO2 * -1.
	count += 1
	sys.stdout.write(f"\riter: {count}, lik: {round(opt_outs_CO2.item(),2)}")
	if (count) % 25 == 0:
		print("")
		print("...")
		print("")
		print(f"Optimized parameters: {guess1}")
		print(f"Objective function value (negative log-likelihood): {round(opt_outs_CO2.item(),2)}")
		print("")
		print("...")
		print("")		
			
	return opt_outs_CO2.item()
	
guess_orig=numpy.array([1.e-02,1.54031956e-07,1.e-01,9.0e+01
,6.0e+00,5.e+02,1.00000000e+00,2.94904337e-01
,1.01544031e+01,1.0e+01,1.0e+00,5.0e-02
,1.88691442e-01,6.77969523e+00])

file_out = wkspc + 'neldermead_hyperparams' + str(sys.argv[1]) + '.csv'

bounds = [(0, numpy.inf) for _ in guess_orig]

bounds[2] = (1.0e-02, numpy.inf)
bounds[3] = (50., numpy.inf)
bounds[4] = (1., numpy.inf)
bounds[5] = (2.5, numpy.inf)
bounds[6] = (0.01,1.)

bounds[11] = (0, 0.1)
bounds[12] = (0, 0.2)

result = minimize(optimize_MLE_merge_guess, guess_orig, method='Nelder-Mead', bounds=bounds)

print("")
print("...")
print("")
print(f"Optimization finished with success: {result.success}")
print(f"Optimized parameters: {result.x}")
print(f"Objective function value (negative log-likelihood): {result.fun}")
	
numpy.savetxt(file_out, result.x.reshape(1, -1), delimiter=',', fmt='%10.8e')