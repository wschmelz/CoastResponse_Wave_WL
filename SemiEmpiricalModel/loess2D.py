import math
import csv
import os
import sys
import numpy
from numpy import genfromtxt
from numpy.fft import fft, ifft, ifftshift
import scipy
from scipy import signal
from scipy.interpolate import interp1d

import time

from datetime import tzinfo, timedelta, datetime


def loess_int(x_interp,y_interp,t_series_x,t_series_y,t_series_z,pt_min_r,dist_r,max_dist_r):
	dataset = numpy.zeros((len(x_interp),3))
	dataset[:,0] = x_interp
	dataset[:,1] = y_interp
	pt_min_r_2 = pt_min_r * 1
	for location in range(0,len(dataset[:,0])):
		new_z = "nan"
		x_est = dataset[location,0]
		y_est = dataset[location,1]
		
		
		x_dist = t_series_x - x_est
		y_dist = t_series_y - y_est
		
		xsqr = x_dist * x_dist
		
		ysqr = y_dist * y_dist
		
		dists_all = numpy.sqrt(xsqr + ysqr)
		
		dist_sorted = numpy.argsort(dists_all)
		
		dist_indices = numpy.where((dists_all<dist_r))[0]
		
		dist_indices = dist_indices
		
		if dists_all[dist_sorted][pt_min_r] > dist_r :
			dist_indices = dist_sorted[0:pt_min_r]
		
		x = x_dist[dist_indices]
		y = y_dist[dist_indices]
		z = t_series_z[dist_indices]
		dists = dists_all[dist_indices] 
		
		dist_max = numpy.max(dists_all[dist_indices])

		if dist_max <= max_dist_r:
			dists_norm = dists/dist_r
			if dists_all[dist_sorted][pt_min_r] > dist_r :
				dists_norm = dists/dist_max
			ones = numpy.ones(len(dist_indices))
			
			xy = x*y
			xx = x*x
			yy = y*y 
			
			A = numpy.transpose(numpy.reshape(numpy.array([(ones), (x), (y), (xy), (xx), (yy)]),(6,-1)))
			
			w = ((1.0-((dists_norm*dists_norm*dists_norm)))*(1.0-((dists_norm*dists_norm*dists_norm)))*(1.0-((dists_norm*dists_norm*dists_norm))))
			w = numpy.reshape(w,(-1,1))
			
			A = A * numpy.repeat(w,6,1)
			b = numpy.reshape(w[:,0]*z,(-1,1))
		
			new_x = numpy.linalg.lstsq(A, b)[0]	
			
			new_z = numpy.reshape(numpy.array([new_x[0]]),(-1,1))
			
			if len(new_x[0]) == 0:
				print ('fail')
				new_x[0] = 0.0			
		if dist_max >= max_dist_r:
			new_z = "nan"
			dist_max = max_dist_r

		
		dataset[location,2] = float(new_z)
		
		sys.stdout.write("\rLOESS Interpolation %i of %i complete: dist: %s points: %i  z: %s             " % (location+1,len(dataset[:,0]),str(round(float(dist_max),2)),len(dist_indices),str(round(float(new_z),2))))
		
	return dataset[:,0], dataset[:,1], dataset[:,2]
	
def weighted_linear(x_interp,y_interp,t_series_x,t_series_y,t_series_z,pt_min_r,dist_r,max_dist_r):
	dataset = numpy.zeros((len(x_interp),3))
	dataset[:,0] = x_interp
	dataset[:,1] = y_interp
	pt_min_r_2 = pt_min_r * 1
	for location in range(0,len(dataset[:,0])):
		new_z = "nan"
		x_est = dataset[location,0]
		y_est = dataset[location,1]
		
		if x_est < 1.0 and x_est > .78:
			pt_min_r = 120
		if x_est > 1.0 or x_est == 1.0:
			pt_min_r = pt_min_r_2 * 1			
		
		x_dist = t_series_x - x_est
		y_dist = t_series_y - y_est
		
		xsqr = x_dist * x_dist
		
		ysqr = y_dist * y_dist
		
		dists_all = numpy.sqrt(xsqr + ysqr)
		
		dist_sorted = numpy.argsort(dists_all)
		
		dist_indices = numpy.where((dists_all<dist_r))[0]
		
		dist_indices = dist_indices
		
		if dists_all[dist_sorted][pt_min_r] > dist_r :
			dist_indices = dist_sorted[0:pt_min_r]
		
		x = x_dist[dist_indices]
		y = y_dist[dist_indices]
		z = t_series_z[dist_indices]
		dists = dists_all[dist_indices] 
		
		dist_max = numpy.max(dists_all[dist_indices])

		if dist_max <= max_dist_r:
			dists_norm = dists/dist_r
			if dists_all[dist_sorted][pt_min_r] > dist_r :
				dists_norm = dists/dist_max			
			ones = numpy.ones(len(dist_indices))
			
			xy = x*y
			xx = x*x
			yy = y*y 
			
			A = numpy.transpose(numpy.reshape(numpy.array([(ones), (x), (y)]),(3,-1)))
			
			w = ((1.0-((dists_norm*dists_norm*dists_norm)))*(1.0-((dists_norm*dists_norm*dists_norm)))*(1.0-((dists_norm*dists_norm*dists_norm))))
			w = numpy.reshape(w,(-1,1))
			
			A = A * numpy.repeat(w,3,1)
			b = numpy.reshape(w[:,0]*z,(-1,1))
		
			new_x = numpy.linalg.lstsq(A, b,rcond=None)[0]	
			
			new_z = numpy.reshape(numpy.array([new_x[0]]),(-1,1))
			
			if len(new_x[0]) == 0:
				print ('fail')
				new_x[0] = 0.0			
		if dist_max >= max_dist_r:
			new_z = "nan"
			dist_max = max_dist_r

		
		dataset[location,2] = float(new_z)
		
		sys.stdout.write("\rLOESS Interpolation %i of %i complete: dist: %s points: %i  z: %s             " % (location+1,len(dataset[:,0]),str(round(float(dist_max),2)),len(dist_indices),str(round(float(new_z),2))))
		
	return dataset[:,0], dataset[:,1], dataset[:,2]	