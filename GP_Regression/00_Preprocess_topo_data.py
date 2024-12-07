import os
import sys
import datetime
import re
import shutil
import time
from time import gmtime, strftime
import csv
import numpy
import random
import glob
import matplotlib.pyplot as plt
import scipy
from scipy import interpolate
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy.stats import linregress

from numpy import matrix
from numpy import genfromtxt
from numpy import linalg
import requests
##dowloaad LOESS algorithm
'''
url = 'https://raw.githubusercontent.com/wschmelz/GeologicalModeling/main/Scripts/loess.py'
loess_py = requests.get(url)  

with open('loess_py.py', 'w') as f:
    f.write(loess_py.text)
''' 

import loess_py
import points_within 
from scipy.optimize import minimize

import sys
print(sys.version)

from numba import njit
import torch
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
'''
device = torch.device("cuda:0" if is_cuda_available else "cpu")
print(f"Using device: {device}")
'''
backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))

print (time.strftime("%H:%M:%S"))

wkspc_data = wkspc + '00_Data/'

gpsdata = glob.glob(wkspc_data + "*/")

print ("")

def project2(xy_array,depths,start_x,start_y,end_x,end_y):
	pts = len(xy_array[:,0])
	log_dists = []
	
	angle = numpy.arctan2((end_y-start_y),(end_x-start_x))
	
	new_xy_array = numpy.zeros((pts,2))
	new_profile_dists = numpy.zeros((pts,1))
	
	for n in range(0,pts):

		log_x = xy_array[n,0]
		log_y = xy_array[n,1]
		
		tx1 = float(log_x)
		ty1 = float(log_y)
		
		tx2 = tx1 + (numpy.cos(angle + (numpy.pi/2.0)))
		ty2 = ty1 + (numpy.sin(angle + (numpy.pi/2.0)))

		if ty2 - ty1 == 0:
			ty2 = ty1 + .0000000001						
		if tx2 - tx1 == 0:
			tx2 = tx1 - .00000000001	
					
		A11 = ty2-ty1
		A12 = -1*(tx2-tx1)
		b11 = -1*((tx2*ty1)-(tx1*ty2))
			
		shox1 = start_x
		shoy1 = start_y
		shox2 = shox1 + (1.0*(10**9) * numpy.cos(angle))
		shoy2 = shoy1 + (1.0*(10**9) * numpy.sin(angle))	

		temp_y = ty1 + ((shox2 - tx1) * numpy.sin(angle))

		if shoy2 - shoy1 == 0:
			shoy2 = shoy1 - .0000000002
		if shox2 - shox1 == 0:
			shox2 = shox1 + .000000001	
					
		A21 = shoy1-shoy2
		A22 = -1*(shox1-shox2)
		b12 = -1*((shox1*shoy2)-(shox2*shoy1))
		A = numpy.array([[A11,A12],[A21,A22]])
		b = numpy.array([[b11],[b12]])
		
		solution = numpy.linalg.solve(A,b)

		solx = float(solution [0,:])
		soly = float(solution [1,:])
		
		log_dist = numpy.sqrt(((solx - shox1)**2) + ((soly - shoy1)**2))
		
		
		new_xy_array[n,0] = solx
		new_xy_array[n,1] = soly
		new_profile_dists[n,0] = log_dist
	
	index = numpy.argsort(new_profile_dists[:,0])
	
	new_profile_dists = numpy.ndarray.flatten(new_profile_dists[:,:])
	new_xy_array = new_xy_array[:,:]
	depths = depths[:]	
	
	return new_xy_array, new_profile_dists, depths, angle


for n in range(0,len(gpsdata)):
	print (gpsdata[n])
	
	gpsdata1 = glob.glob(gpsdata[n] + "01_GCPs/" + "*.csv")
	gpsdata2 = glob.glob(gpsdata[n] + "02_Transects/" + "*.csv")
	gpsdata3 = glob.glob(gpsdata[n] + "03_DEMs/" + "*.csv")
	
	print (gpsdata1[0])
	print (gpsdata2[0])
	print (gpsdata3[0])
	time_tmp_orig = gpsdata2[0].split("_")[-1][0:-4]
	
	year_tmp = int(time_tmp_orig[0:4])
	month_tmp = int(time_tmp_orig[4:6])
	day_tmp = int(time_tmp_orig[6:])
	
	if n == 0:
		time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp)
		
		time_tmp = (time_datum_start - time_datum_start).total_seconds() / (3600.0*24.)
		
	if n > 0:
		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp)	
	
		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)

	print(time_tmp)

	GCP_data = numpy.genfromtxt(gpsdata1[0],delimiter=',')
	
	topo_data_1 = numpy.zeros((len(GCP_data[:,0]),6))
	topo_data_1[:,0] = time_tmp
	topo_data_1[:,1] = GCP_data[:,1]
	topo_data_1[:,2] = GCP_data[:,2]
	topo_data_1[:,3] = GCP_data[:,3]
	topo_data_1[:,4] = (topo_data_1[:,4] * 0.0) + 1.0

	transect_data = numpy.genfromtxt(gpsdata2[0],delimiter=',')
	
	topo_data_2 = numpy.zeros((len(transect_data[:,0]),6))
	topo_data_2[:,0] = time_tmp
	topo_data_2[:,1] = transect_data[:,1]
	topo_data_2[:,2] = transect_data[:,2]
	topo_data_2[:,3] = transect_data[:,3]
	topo_data_2[:,4] = (topo_data_2[:,4] * 0.0) + 0.0
	topo_data_2[:,5] = transect_data[:,0]
	
	DEM_data = numpy.genfromtxt(gpsdata3[0],delimiter=',')
	
	topo_data_3 = numpy.zeros((len(DEM_data[::4,0]),6))
	topo_data_3[:,0] = time_tmp
	topo_data_3[:,1] = DEM_data[::4,0]
	topo_data_3[:,2] = DEM_data[::4,1]
	topo_data_3[:,3] = DEM_data[::4,2]
	topo_data_3[:,4] = (topo_data_3[:,4] * 0.0) + 2.0	
	
	topo_data_4 = numpy.append(topo_data_1,topo_data_2,axis=0)
	topo_data_4 = numpy.append(topo_data_4,topo_data_3,axis=0)
	
	if n == 0:
	
		topo_data_combined_orig = topo_data_4 * 1.0	
	
	if n > 0:
		topo_data_combined_orig = numpy.append(topo_data_combined_orig,topo_data_4,axis=0)
	
	print ("")
	
print ("")

file_out = wkspc + 'all_data.csv'	
numpy.savetxt(file_out,topo_data_combined_orig,delimiter=',',fmt='%10.8e')

xmin = numpy.floor(numpy.min(topo_data_combined_orig[:,1]))
xmax = numpy.ceil(numpy.max(topo_data_combined_orig[:,1]))
ymin = numpy.floor(numpy.min(topo_data_combined_orig[:,2]))
ymax = numpy.ceil(numpy.max(topo_data_combined_orig[:,2]))

cellsize_x = 2.5
cellsize_y = 2.5

xi = numpy.arange(xmin, xmax+cellsize_x,cellsize_x)
yi = numpy.arange(ymin, ymax+cellsize_y,cellsize_y)

X, Y = numpy.meshgrid(xi, yi)

x_flat = numpy.ndarray.flatten(X)
y_flat = numpy.ndarray.flatten(Y)

mask = numpy.ones(numpy.shape(X))
count = 0
count2 = 0

w1 = numpy.where(topo_data_combined_orig[:,4]==0)[0]
transect_data2 = topo_data_combined_orig[w1,:]

t_pts = numpy.genfromtxt(wkspc + 'Transect_pts.csv',delimiter=',')

fig = plt.figure(1,figsize=(12,15))

colors = ["red","orangered","yellow","chartreuse","deepskyblue","mediumblue","indigo","violet","black"]

for n in range(1,6):

	w2 = numpy.where(transect_data2[:,5]==n)[0]
	w3 = numpy.where(t_pts[:,0]==n)[0]
	
	output = project2(transect_data2[w2,1:3],transect_data2[w2,3],t_pts[w3[0],1],t_pts[w3[0],2],t_pts[w3[1],1],t_pts[w3[1],2])
	
	new_dist_2 = numpy.arange(0,numpy.max(output[1]),1.0)
	
	out_matrix_tmp = numpy.zeros((len(new_dist_2),5)) * numpy.nan
	out_matrix_tmp[:,0] = new_dist_2
	
	new_xes = t_pts[w3[0],1] + (new_dist_2 * numpy.cos(output[3]))
	new_yes = t_pts[w3[0],2] + (new_dist_2 * numpy.sin(output[3]))
	ax1 = plt.subplot(5,1,n)
	for n2 in range(0,len(gpsdata)):
		out_matrix_tmp = numpy.zeros((len(new_dist_2),5)) * numpy.nan
		out_matrix_tmp[:,0] = new_dist_2
		
		gpsdata1 = glob.glob(gpsdata[n2] + "02_Transects/" + "*.csv")
		time_tmp_orig = gpsdata1[0].split("_")[-1][0:-4]
		
		year_tmp = int(time_tmp_orig[0:4])
		month_tmp = int(time_tmp_orig[4:6])
		day_tmp = int(time_tmp_orig[6:])
		
		label1 = str(month_tmp).zfill(2) + "/" + str(day_tmp).zfill(2) + "/" + str(year_tmp)
		
		if n2 == 0:
			time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp)
			
			time_tmp = (time_datum_start - time_datum_start).total_seconds() / (3600.0*24.)
			
		if n2 > 0:
			time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp)	
		
			time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)

		w4 = numpy.where(transect_data2[w2,0]==time_tmp)[0]

		w10 = numpy.where(output[1][w4]<0.05)[0]
		
		output[1][w4[w10]] = output[1][w4[w10]] - numpy.min(output[1][w4])
		
		f = interpolate.interp1d(output[1][w4], output[2][w4])
		
		w5 = numpy.where((new_dist_2>=numpy.min(output[1][w4]))&(new_dist_2<=numpy.max(output[1][w4])))[0]

		new_output = f(new_dist_2[w5])
		
		out_matrix_tmp[:,0] = time_tmp
		out_matrix_tmp[:,1] = new_xes
		out_matrix_tmp[:,2] = new_yes
		out_matrix_tmp[w5,3] = new_output
		out_matrix_tmp[:,4] = 0.0

		if n2 > 0:
			w6 = numpy.where((new_dist_2<numpy.min(output[1][w4])))[0]
			if len(w6)>0:
				for n3 in range(0,len(w6)):
					w9 = numpy.where((new_xes[w6[n3]]==out_matrix_trans_tmp[:,1])& (new_yes[w6[n3]]==out_matrix_trans_tmp[:,2]))[0]
					out_matrix_tmp[[w6[n3]],3] = numpy.nanmean(out_matrix_trans_tmp[w9,3])
		
		ax1.plot(new_dist_2,out_matrix_tmp[:,3],linewidth=1.25,color=colors[n2],label=label1)
		

		if n2 == 0:
			out_matrix_trans_tmp = out_matrix_tmp * 1.0
		if n2 >0:
			out_matrix_trans_tmp = numpy.append(out_matrix_trans_tmp,out_matrix_tmp,axis=0)
	if n==1:
		ax1.legend(fontsize=7)
	
	ax1.plot([0,1000],[0,0],linewidth=1.,color="dodgerblue")
	
	profile_label = "LBT " + str(n)
	ax1.set_title(profile_label)		

	ax1.set_xlim(0,160)
	ax1.set_ylim(-.5,7.5)
	ax1.set_ylabel("Elevation (m rel. to NAVD88)")
	if n==5:
		ax1.set_xlabel("Profile distance (m)")
	ax1.grid()
		
	if n == 1:
		out_matrix_trans = out_matrix_tmp * 1.0
	if n > 1:
		out_matrix_trans = numpy.append(out_matrix_trans,out_matrix_trans_tmp,axis=0)
plt.tight_layout()
pltname = wkspc + 'LBT_2D.png'
plt.savefig(pltname, dpi = 300)
plt.close()


@njit
def compute_mask(X, Y, topo_data_combined_orig):
	mask = numpy.ones(numpy.shape(X))
	count = 0
	count2 = 0

	for n1 in range(0, X.shape[0]):
		for n2 in range(0, X.shape[1]):
			count2 += 1
			if numpy.min(numpy.sqrt((X[n1, n2] - topo_data_combined_orig[::10, 1]) ** 2 + (Y[n1, n2] - topo_data_combined_orig[::10, 2]) ** 2)) > 5.0:
				mask[n1, n2] = numpy.nan
				count2 -= 1
			count += 1
			if (count % 10000) == 0:
				print("total pts:", count2, "; completed", count, "of", X.shape[0] * X.shape[1])
				
	return mask

@njit
def compute_mask2(X, Y, topo_data_3,Z_isopach):
	
	count = 0
	count2 = 0

	for n1 in range(0,len(X[:,0])):
		for n2 in range(0,len(X[0,:])):
			if numpy.isnan(Z_isopach[n1,n2]) == False:
				count2 = count2+1
				if numpy.min(numpy.sqrt(((X[n1,n2]-topo_data_3[::2,1])**2.) + ((Y[n1,n2]-topo_data_3[::2,2])**2.)))>2.5:
					Z_isopach[n1,n2] = numpy.nan
					count2 = count2-1
			count = count+1
			if (count % 10000) == 0:
				print("total pts:", count2, "; completed", count, "of", X.shape[0] * X.shape[1])
	return Z_isopach

mask = compute_mask(X, Y, topo_data_combined_orig)

print ("")
for n in range(0,len(gpsdata)):
	print (gpsdata[n])
	
	gpsdata1 = glob.glob(gpsdata[n] + "01_GCPs/" + "*.csv")
	gpsdata2 = glob.glob(gpsdata[n] + "02_Transects/" + "*.csv")
	gpsdata3 = glob.glob(gpsdata[n] + "03_DEMs/" + "*.csv")
	
	print (gpsdata1[0])
	print (gpsdata2[0])
	print (gpsdata3[0])

	time_tmp_orig = gpsdata2[0].split("_")[-1][0:-4]
	
	year_tmp = int(time_tmp_orig[0:4])
	month_tmp = int(time_tmp_orig[4:6])
	day_tmp = int(time_tmp_orig[6:])
	
	if n == 0:
		time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp)
		
		time_tmp = (time_datum_start - time_datum_start).total_seconds() / (3600.0*24.)
		
	if n > 0:
		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp)	
	
		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)

	print(time_tmp)

	GCP_data = numpy.genfromtxt(gpsdata1[0],delimiter=',')
	
	topo_data_1 = numpy.zeros((len(GCP_data[:,0]),6))
	topo_data_1[:,0] = time_tmp
	topo_data_1[:,1] = GCP_data[:,1]
	topo_data_1[:,2] = GCP_data[:,2]
	topo_data_1[:,3] = GCP_data[:,3]
	topo_data_1[:,4] = (topo_data_1[:,4] * 0.0) + 1.0
	topo_data_1[:,5] = GCP_data[:,0]*0.0

	transect_data = numpy.genfromtxt(gpsdata2[0],delimiter=',')
	
	topo_data_2 = numpy.zeros((len(transect_data[:,0]),6))
	topo_data_2[:,0] = time_tmp
	topo_data_2[:,1] = transect_data[:,1]
	topo_data_2[:,2] = transect_data[:,2]
	topo_data_2[:,3] = transect_data[:,3]
	topo_data_2[:,4] = (topo_data_2[:,4] * 0.0) + 0.0
	topo_data_2[:,5] = transect_data[:,0]
	
	DEM_data = numpy.genfromtxt(gpsdata3[0],delimiter=',')
	
	topo_data_3 = numpy.zeros((len(DEM_data[::2,0]),6))
	topo_data_3[:,0] = time_tmp
	topo_data_3[:,1] = DEM_data[::2,0]
	topo_data_3[:,2] = DEM_data[::2,1]
	topo_data_3[:,3] = DEM_data[::2,2]
	topo_data_3[:,4] = (topo_data_3[:,4] * 0.0) + 2.0	
		
	fig = plt.figure(1,figsize=(8,10))

	ax1 = plt.subplot(1,1,1)
			
	cmap1 = plt.get_cmap('gist_ncar')
	cmap1.set_under('w')
	
	Z_isopach = griddata(topo_data_3[:,1:3], topo_data_3[:,3], (X, Y), method='cubic',fill_value=numpy.nan) * mask

	count = 0
	count2 = 0
	Z_isopach = compute_mask2(X, Y, topo_data_3,Z_isopach)	

	print ("")
	
	p1 = ax1.pcolormesh(X,Y,Z_isopach,vmin=0,vmax=10,cmap=cmap1)
	ax1.set_aspect('equal',adjustable='box')
	ax1.set_ylabel("Northing")
	ax1.set_xlabel("Easting")	
	

	ax1.set_xlim(xmin,xmax)	
	ax1.set_ylim(ymin,ymax)	
	
	ax1.grid(linestyle='-.',linewidth=0.15)	
	
	cbar = plt.colorbar(p1,ax=ax1,shrink=0.75)
	cbar.set_label('Elevation (m)')
	
	pltname = wkspc + "LBT_3D_" + str(time_tmp_orig) + '.png'

	plt.tight_layout()
	plt.savefig(pltname, dpi = 300)
	plt.close()
	
	z_vals = numpy.ndarray.flatten(Z_isopach)
	w_true = numpy.where(numpy.isnan(z_vals)==False)[0]
	
	topo_data_4 = numpy.zeros((len(w_true),6))
	topo_data_4[:,0] = topo_data_4[:,0] + time_tmp
	topo_data_4[:,1] = x_flat[w_true]
	topo_data_4[:,2] = y_flat[w_true]
	topo_data_4[:,3] = z_vals[w_true]
	topo_data_4[:,4] = (topo_data_4[:,4] * 0.0) + 2.0		
	
	
	topo_data_5 = numpy.append(topo_data_1,topo_data_2,axis=0)
	topo_data_5 = numpy.append(topo_data_5,topo_data_4,axis=0)
	
	if n == 0:
	
		topo_data_combined = topo_data_5 * 1.0	
	
	if n > 0:
		topo_data_combined = numpy.append(topo_data_combined,topo_data_5,axis=0)

file_out = wkspc + 'processed_data.csv'	
numpy.savetxt(file_out,topo_data_combined,delimiter=',',fmt='%10.8e')

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

file_out = wkspc + 'processed_data.csv'	
topo_data_combined = numpy.genfromtxt(file_out,delimiter=',')

w1 = numpy.where(numpy.isnan(topo_data_combined[:,3]))[0]
topo_data_combined = numpy.delete(topo_data_combined,w1,axis=0)

w1_tmp = ((topo_data_combined[:,0]==17.) & (topo_data_combined[:,4]==2.))
w1 = numpy.where((w1_tmp==True) & (topo_data_combined[:,2]>4.3771*(10.**6.)))[0]
topo_data_combined = numpy.delete(topo_data_combined,w1,axis=0)

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

msk_wkspc = wkspc + '01_Masks/'
file_out = msk_wkspc + 'Exclusion_pts.csv'	
comp_bound = numpy.genfromtxt(file_out,delimiter=',')

mask_flat = points_within.points_mask(x_flat, y_flat, comp_bound)

x_new = x_flat[mask_flat]
y_new = y_flat[mask_flat]

x,y,z = topo_data_combined[:,1],topo_data_combined[:,2],topo_data_combined[:,3]

Z_tmp = numpy.zeros(numpy.shape(X))
Z_flat = numpy.ndarray.flatten(Z_tmp)

pt_min_r = 10
dist_r = 5.
factor = 2

Z1_loess = loess_py.loess_2D(x_new,y_new,x,y,z,pt_min_r,dist_r,factor)

w_0 = numpy.where(Z1_loess<=0)[0]
Z1_loess[w_0] = 0.0

Z_flat[mask_flat] = Z1_loess

output_tmp = numpy.zeros((len(x_flat),3))

output_tmp[:,0] = x_flat
output_tmp[:,1] = y_flat
output_tmp[:,2] = Z_flat

file_out = wkspc + 'processed_data_LOESS_0.csv'	
numpy.savetxt(file_out,output_tmp,delimiter=',',fmt='%10.8e')

Z1_loess = numpy.reshape(Z_flat,numpy.shape(X))

fig = plt.figure(1,figsize=(8,10))

ax1 = plt.subplot(1,1,1)
		
cmap1 = plt.get_cmap('gist_ncar')
cmap1.set_under('w')

p1 = ax1.pcolormesh(X,Y,Z1_loess,vmin=0,vmax=10,cmap=cmap1)
ax1.set_aspect('equal',adjustable='box')
ax1.set_ylabel("Northing")
ax1.set_xlabel("Easting")	


ax1.set_xlim(xmin,xmax)	
ax1.set_ylim(ymin,ymax)	

ax1.grid(linestyle='-.',linewidth=0.15)	

cbar = plt.colorbar(p1,ax=ax1,shrink=0.75)
cbar.set_label('Elevation (m)')

pltname = wkspc + "LBT_3D_all.png"

plt.tight_layout()
plt.savefig(pltname, dpi = 300)

file_out = msk_wkspc + 'Exclusion_pts2.csv'	
comp_bound2 = numpy.genfromtxt(file_out,delimiter=',')

topo_data_combined_mask = points_within.points_mask(topo_data_combined[:, 1], topo_data_combined[:, 2], comp_bound2)

topo_data_combined = topo_data_combined[topo_data_combined_mask,:] * 1.0

x,y,z = topo_data_combined[:,1],topo_data_combined[:,2],topo_data_combined[:,3]

z_loess_interpolated = griddata((x_flat, y_flat), numpy.ndarray.flatten(Z1_loess), (x, y), method='linear')

wz_0 = numpy.where(z<-0.1)[0]

z_corrected = z - z_loess_interpolated

topo_data_combined[:, 3] = z_corrected

topo_data_combined_mask2 = topo_data_combined[:, 3]>0

output_tmp = numpy.zeros((len(x),6))

output_tmp[:,0] = topo_data_combined[:,0]
output_tmp[:,1] = x
output_tmp[:,2] = y
output_tmp[:,3] = z
output_tmp[:,4] = z_loess_interpolated
output_tmp[:,5] = z_corrected

output_tmp = numpy.delete(output_tmp,wz_0,axis=0)

file_out = wkspc + 'processed_data_LOESS_1.csv'	
numpy.savetxt(file_out,output_tmp,delimiter=',',fmt='%10.8e')

topo_data_combined = numpy.delete(topo_data_combined,wz_0,axis=0)

w1 = numpy.where(numpy.isnan(topo_data_combined[:,3]))[0]
topo_data_combined = numpy.delete(topo_data_combined,w1,axis=0)

file_out = wkspc + 'processed_data_LOESS.csv'	
numpy.savetxt(file_out,topo_data_combined,delimiter=',',fmt='%10.8e')