import os
import sys
import time
from time import strftime
import numpy
import glob
import matplotlib.pyplot as plt

import scipy
from scipy import interpolate
from scipy.interpolate import griddata

import loess_py
import points_within 

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))

print (time.strftime("%H:%M:%S"))

wkspc_data = wkspc + '00_Data/'
msk_wkspc = wkspc + '01_Masks/'

gpsdata = glob.glob(wkspc_data + "*/")

print ("")

file_out = wkspc + 'all_data.csv'	
topo_data_combined_orig = numpy.genfromtxt(file_out,delimiter=',')

xmin1 = numpy.floor(numpy.min(topo_data_combined_orig[:,1]))
xmax1 = numpy.ceil(numpy.max(topo_data_combined_orig[:,1]))
ymin1 = numpy.floor(numpy.min(topo_data_combined_orig[:,2]))
ymax1 = numpy.ceil(numpy.max(topo_data_combined_orig[:,2]))

##

output_matrix1 = wkspc + 'MC_probability_output_1.csv'	
output_matrix1 = numpy.genfromtxt(output_matrix1,delimiter=',')

##

output_matrix2 = wkspc + 'MC_probability_output_2.csv'	
output_matrix2 = numpy.genfromtxt(output_matrix2,delimiter=',')

##

output_matrix3 = wkspc + 'MC_probability_output_3.csv'	
output_matrix3 = numpy.genfromtxt(output_matrix3,delimiter=',')

##

output_matrix4 = wkspc + 'MC_probability_output_4.csv'	
output_matrix4 = numpy.genfromtxt(output_matrix4,delimiter=',')

##

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
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing) & (topo_data_combined[:, 2] < current_northing + 20.0))[0]

	if in_window.size > 0:
		
		percentile_97_value = numpy.percentile(topo_data_combined[in_window, 1], 99.5)

		new_easting = percentile_97_value
		zero_val = 0.04 * numpy.random.randn()
		new_point = [new_easting, current_northing + 10.0]
		new_points.append(new_point)

	current_northing += 20.0

current_northing = max_northing
while current_northing >= min_northing:
   
	in_window = numpy.where((topo_data_combined[:, 2] >= current_northing - 20.0) & (topo_data_combined[:, 2] < current_northing))[0]

	if in_window.size > 0:
		
		percentile_97_value = numpy.percentile(topo_data_combined[in_window, 1], 1)

		new_easting = percentile_97_value
		zero_val = 0.04 * numpy.random.randn()
		new_point = [new_easting, current_northing + 10.0]
		new_points.append(new_point)

	current_northing -= 20.0

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

w_high = numpy.where(numpy.abs(topo_data_combined[:,3] > 3.0))[0]
topo_data_combined = numpy.delete(topo_data_combined,w_high,axis=0)

x_mean = numpy.mean(topo_data_combined[:,1])
y_mean = numpy.mean(topo_data_combined[:,2])											
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

cellsize_x = 2.0
cellsize_y = 2.0

xi = numpy.arange(xmin, xmax+cellsize_x,cellsize_x)
yi = numpy.arange(ymin, ymax+cellsize_y,cellsize_y)

X, Y = numpy.meshgrid(xi, yi)

x_flat = numpy.ndarray.flatten(X)
y_flat = numpy.ndarray.flatten(Y)

file_out = msk_wkspc + 'Exclusion_pts2.csv'	
comp_bound = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(x_flat , y_flat, new_points_array1)
mask = numpy.reshape(mask_flat,numpy.shape(X))

x_new = x_flat[mask_flat]
y_new = y_flat[mask_flat]

location_array_tmp = numpy.zeros((len(output_matrix3),2))
location_array_tmp[:,0] = output_matrix3
location_array_tmp[:,1] = output_matrix4

time_array_tmp2 = numpy.zeros(len(location_array_tmp[:,0]))

isopach = numpy.zeros((len(location_array_tmp[:,1]),1)) * numpy.nan

location_array_tmp_real = numpy.zeros((len(location_array_tmp[:,1]),2)) * numpy.nan
location_array_tmp_real[:,0] = location_array_tmp[:,0] + x_mean
location_array_tmp_real[:,1] = location_array_tmp[:,1] + y_mean															   
file_out1 = wkspc + 'processed_data_LOESS_0.csv'	
topo_data_LOESS = numpy.genfromtxt(file_out1,delimiter=',')

z_fill = griddata(topo_data_LOESS[:,0:2], topo_data_LOESS[:,2], topo_data_combined[:,1:3], method='linear',fill_value=numpy.nan)
topo_data_combined[:,3] = topo_data_combined[:,3] + z_fill

for n in range(0,len(time_array_tmp)):

	gpsdata1 = glob.glob(gpsdata[n] + "02_Transects/" + "*.csv")
	time_tmp_orig = gpsdata1[0].split("_")[-1][0:-4]
	
	fig = plt.figure(1,figsize=(8,10))

	ax1 = plt.subplot(1,1,1)
			
	cmap1 = plt.get_cmap('gist_ncar')
	cmap1.set_under('w')

	w_time = numpy.where(output_matrix2 == time_array_tmp[n])[0]
	Z_isopach1 = griddata(location_array_tmp_real[w_time,0:2], output_matrix1[w_time], (X, Y), method='linear',fill_value=numpy.nan) * mask
	Z_isopach2 = griddata(topo_data_LOESS[:,0:2], topo_data_LOESS[:,2], (X, Y), method='linear',fill_value=numpy.nan) * mask
	
	Z_isopach = Z_isopach1 + Z_isopach2
	
	count = 0
	count2 = 0
	
	x_flat1 = numpy.ndarray.flatten(X)
	y_flat1 = numpy.ndarray.flatten(Y)
	z_flat1 = numpy.ndarray.flatten(Z_isopach)
	
	w_0 = numpy.where((numpy.isnan(z_flat1)==False)&(z_flat1>0))[0]
	w_t = numpy.where(topo_data_combined[:,0] == time_array_tmp[n])[0] 
	new_points = []
	for n1 in range(len(w_0)):
		
		x_w0 = x_flat1[w_0[n1]]
		y_w0 = y_flat1[w_0[n1]]
		z_w0 = z_flat1[w_0[n1]]
		
		dist = numpy.sqrt(((x_w0-topo_data_combined[w_t,1])**2.) + ((y_w0-topo_data_combined[w_t,2])**2.))
		w_dists = numpy.where(dist<2.) [0]
		if len(w_dists)==0:
			new_point = [time_array_tmp[n], x_w0, y_w0, z_w0, 1.0,0.0]
			new_points.append(new_point)
	topo_interp = topo_data_combined[w_t,:]*1.0
	if len(new_points) > 1:
		new_points = numpy.array(new_points)
		topo_interp = numpy.append(topo_interp,new_points,axis=0)
	
	x,y,z = topo_interp[:,1],topo_interp[:,2],topo_interp[:,3]

	pt_min_r = 10

	dist_r = 10.

	factor = 2

	Z1_loess = loess_py.loess_2D(x_flat1[w_0],y_flat1[w_0],x,y,z,pt_min_r,dist_r,factor)
		
	topo_data_GPR_tmp = numpy.zeros((len(x_flat1[w_0]),5)) * numpy.nan
	topo_data_GPR_tmp[:,0] = time_array_tmp[n]
	topo_data_GPR_tmp[:,1] = x_flat1[w_0]
	topo_data_GPR_tmp[:,2] = y_flat1[w_0]
	topo_data_GPR_tmp[:,3] = Z1_loess
	topo_data_GPR_tmp[:,4] = (x_flat1[w_0] * 0.0) + 1.0	
	
	if n == 0:
		topo_data_GPR = topo_data_GPR_tmp * 1.0
	if n > 0:
		topo_data_GPR = numpy.append(topo_data_GPR,topo_data_GPR_tmp,axis=0)
	
	w_1 = numpy.where((numpy.isnan(Z_isopach)==False)&(Z_isopach<=0))
	Z_isopach[w_1[0],w_1[1]] = Z_isopach[w_1[0],w_1[1]] * 0.0
	p1 = ax1.pcolormesh(X,Y,Z_isopach,vmin=0,vmax=10,cmap=cmap1)
	ax1.set_aspect('equal',adjustable='box')
	ax1.set_ylabel("Northing")
	ax1.set_xlabel("Easting")	

	ax1.set_xlim(xmin1,xmax1)	
	ax1.set_ylim(ymin1,ymax1)	
	
	ax1.grid(linestyle='-.',linewidth=0.15)	
	
	cbar = plt.colorbar(p1,ax=ax1,shrink=0.75)
	cbar.set_label('Elevation (m)')
	
	pltname = wkspc + "LBT_3D_" + str(time_tmp_orig) + '_GP_Reg.png'

	plt.tight_layout()
	plt.savefig(pltname, dpi = 300)
	plt.close()

	fig = plt.figure(1,figsize=(8,10))

	ax1 = plt.subplot(1,1,1)
			
	cmap1 = plt.get_cmap('gist_ncar')
	cmap1.set_under('w')
	
	Z_isopach1 = griddata(topo_data_GPR_tmp[:,1:3], topo_data_GPR_tmp[:,3], (X, Y), method='linear',fill_value=numpy.nan) * mask
	w_1 = numpy.where((numpy.isnan(Z_isopach1)==False)&(Z_isopach1<=0))
	Z_isopach1[w_1[0],w_1[1]] = Z_isopach1[w_1[0],w_1[1]] * 0.0	
	p1 = ax1.pcolormesh(X,Y,Z_isopach1,vmin=0,vmax=10,cmap=cmap1)
	ax1.set_aspect('equal',adjustable='box')
	ax1.set_ylabel("Northing")
	ax1.set_xlabel("Easting")	

	ax1.set_xlim(xmin1,xmax1)	
	ax1.set_ylim(ymin1,ymax1)	
	
	ax1.grid(linestyle='-.',linewidth=0.15)	
	
	cbar = plt.colorbar(p1,ax=ax1,shrink=0.75)
	cbar.set_label('Elevation (m)')
	
	pltname = wkspc + "LBT_3D_" + str(time_tmp_orig) + '_final.png'

	plt.tight_layout()
	plt.savefig(pltname, dpi = 300)
	plt.close()
file_out = wkspc + 'processed_data_GPR.csv'	
numpy.savetxt(file_out,topo_data_GPR,delimiter=',',fmt='%10.8e')