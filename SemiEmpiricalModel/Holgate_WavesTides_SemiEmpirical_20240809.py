import sys
import os
import time
import numpy
import glob
import datetime
import MCMC_20230607
import matplotlib.pyplot as plt
from numba import njit

from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import AnchoredText

print(sys.version)

# Define paths
backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))
print (time.strftime("%H:%M:%S"))

vol_wkspc = wkspc + '02_TopoData/'

# Import variables
tide_file = wkspc + 'tide_output.csv'
tide_output = numpy.genfromtxt(tide_file, delimiter=',')
wave_file = wkspc + 'wave_output.csv'
wave_output = numpy.genfromtxt(wave_file, delimiter=',')

date_list_file = wkspc + 'date_list.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]

date_list = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

#First survey
year_tmp = 2020
month_tmp = 10
day_tmp = 17
hour_tmp = 0

time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp,hour_tmp)

#last survey
year_tmp = 2021
month_tmp = 6
day_tmp = 1
hour_tmp = 0

time_datum_last_data = datetime.datetime(year_tmp, month_tmp, day_tmp,hour_tmp)

#first date
year_tmp = 2020
month_tmp = 1
day_tmp = 1
hour_tmp = 0

time_datum_first_data = datetime.datetime(year_tmp, month_tmp, day_tmp,hour_tmp)

# Define relevant time period
time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)

w_time = numpy.where((tide_output[:, 0] >= 0) & (tide_output[:, 0] < time_tmp_last))[0]

dt = numpy.mean(tide_output[1:,0] - tide_output[0:-1,0])

# ----------------------------- Semi-empirical functions -----------------------------

@njit
def semi_empirical_1(theta, data_series1, dt):
    
    reim_sum = numpy.cumsum(theta[0] * (data_series1[w_time] - theta[1])) * dt
    return reim_sum
	
@njit
def semi_empirical_2(theta, data_series1, mean_val, dt):
    
    T_0 = numpy.zeros_like(data_series1)
    T_0[0] = mean_val

    for si_index in range(1, len(data_series1)):
        T_0[si_index] = T_0[si_index-1] + ((data_series1[si_index] - T_0[si_index-1]) / numpy.absolute(theta[1]))

    reim_sum = numpy.cumsum(theta[0] * (data_series1[w_time] - T_0[w_time])) * dt
    return reim_sum, T_0[w_time]

# ----------------------------- Import survey data -----------------------------

# Read volumes
file_out = vol_wkspc + 'LBT_volumes.txt'
volumes_tmp = numpy.genfromtxt(file_out, delimiter=',')
volumes = volumes_tmp.copy()

# Establish survey dates
gpsdata1 = glob.glob(vol_wkspc + "00_Transects/" + "*.csv")
survey_volumes = numpy.zeros((len(gpsdata1), 5))

date_list_vols = []
w_output = numpy.zeros(len(gpsdata1), dtype="int")

for n, gpsdata in enumerate(gpsdata1):
    time_tmp_orig = gpsdata.split("_")[-1][0:-4]
    year_tmp, month_tmp, day_tmp = int(time_tmp_orig[0:4]), int(time_tmp_orig[4:6]), int(time_tmp_orig[6:])
    
    time_datum = datetime.datetime(year_tmp, month_tmp, day_tmp)
    survey_volumes[n, 0] = (time_datum - time_datum_start).total_seconds() / (3600.0*24.0)
    
    survey_volumes[n, 1:] = [numpy.mean(volumes[:, n])/25., numpy.mean(volumes[0:25, n])/25.,
                             numpy.mean(volumes[25:50, n])/25., numpy.mean(volumes[50:, n])/25.]
    
    date_list_vols.append([time_datum,])
    w_output[n] = numpy.where(tide_output[w_time, 0] == survey_volumes[n, 0])[0]

survey_volumes[:, 1] = numpy.cumsum(survey_volumes[:, 1])
mean_error = 5.65
print("Error:", mean_error)

# Setup parameters
theta_guess = numpy.array ([-0.01,     numpy.mean(tide_output[:,1]), -0.01, 24.*14.,-0.01, 24.*14., -0.01, 24.*14., -0.00, 24.*14.,mean_error])
theta_priors = numpy.array([3.5, numpy.mean(tide_output[:, 1])/10., 3.5, 24.*14., 0.5, 24.*14., 0.5, 24.*14., 0.5, 24.*14., mean_error])
stepsizes = numpy.array([0.1, 0.1*.3048, 1., 50., 1., 50., 1., 50., 1., 50., 0.0]) /3.75

# Function for MCMC calculation
def function(theta):

	theta1 = theta[0:2]
	theta2 = theta[2:4]
	theta3 = theta[4:6]
	theta4 = theta[6:8]
	theta5 = theta[8:10]
	
	semi_empirical1 = semi_empirical_1(theta1,tide_output[:,1],dt)
	semi_empirical2 = semi_empirical_2(theta2,tide_output[:,1],numpy.mean(tide_output[:,1]),dt)
	semi_empirical3 = semi_empirical_2(theta3,wave_output[:,1],numpy.mean(wave_output[:,1]),dt)
	semi_empirical4 = semi_empirical_2(theta4,wave_output[:,2],numpy.mean(wave_output[:,2]),dt)	
	semi_empirical5 = semi_empirical_2(theta5,wave_output[:,3],numpy.mean(wave_output[:,3]),dt)

	semi_empirical = semi_empirical1 + semi_empirical2[0] + semi_empirical3[0] + semi_empirical4[0] + semi_empirical5[0]
	
	return semi_empirical[w_output],semi_empirical,semi_empirical2[1],semi_empirical1+semi_empirical2[0],theta1[1],semi_empirical3[0]+semi_empirical4[0]+semi_empirical5[0],semi_empirical4,semi_empirical5    

# MCMC
training_data = survey_volumes[:, 1].copy()
MCMC_iters, burn_in = 200000, 50000

output_matrix_A,output_matrix_A2,output_matrix_A3,output_matrix_A4,output_matrix_A5,output_matrix_A6,loglik_output = MCMC_20230607.MCMC(function,training_data,theta_guess,theta_priors,stepsizes,MCMC_iters)

# Save results
filename_base = wkspc + '03_Model_output/'
numpy.savetxt(filename_base + "posterior_params.txt", output_matrix_A, delimiter=',')
numpy.savetxt(filename_base + "loglik_output.txt", loglik_output, delimiter=',')

# ----------------------------- Plot results -----------------------------

# Create figure
fig = plt.figure(2,figsize=(10,15))

# ---------------------------- Water level subplot ----------------------------

ax1 = plt.subplot(511)

conv_ft = 1.

# Calculate mean, quantiles for water level variables
T_0_mean = numpy.mean(output_matrix_A3[burn_in:,:],axis=0) * conv_ft
T_0_high2 = numpy.quantile(output_matrix_A3[burn_in:,:],0.975,axis=0) * conv_ft
T_0_low2 = numpy.quantile(output_matrix_A3[burn_in:,:],0.025,axis=0) * conv_ft
T_0_high = numpy.quantile(output_matrix_A3[burn_in:,:],0.16,axis=0) * conv_ft
T_0_low = numpy.quantile(output_matrix_A3[burn_in:,:],0.84,axis=0) * conv_ft

a2_mean = numpy.mean(output_matrix_A5[burn_in:,:],axis=0) * conv_ft
a2_high2 = numpy.quantile(output_matrix_A5[burn_in:,:],0.975,axis=0) * conv_ft
a2_low2 = numpy.quantile(output_matrix_A5[burn_in:,:],0.025,axis=0) * conv_ft
a2_high = numpy.quantile(output_matrix_A5[burn_in:,:],0.16,axis=0) * conv_ft
a2_low = numpy.quantile(output_matrix_A5[burn_in:,:],0.84,axis=0) * conv_ft

q_95 = numpy.quantile(tide_output[:,1],.975) * conv_ft

# Plot water level results
date_list = date_list[w_time]
ax1.plot(date_list,tide_output[w_time,1] * conv_ft,linewidth=1.,color="dodgerblue",label="Atlantic City WL")
label1 = "97.5%ile: " + str(numpy.round(q_95,1)) + " m"
ax1.plot([date_list[0],date_list[-1]],[q_95,q_95],linewidth=1.,color="red",label=label1)

ax1.plot(date_list,numpy.zeros(len(date_list)) + a2_mean,linewidth=1.5,color="maroon",label="$WL_{1}$")
ax1.plot(date_list,numpy.zeros(len(date_list)) + a2_high2,linewidth=0.5,linestyle=":",color="maroon")
ax1.plot(date_list,numpy.zeros(len(date_list)) + a2_high,linewidth=1.0,linestyle="--",color="maroon")
ax1.plot(date_list,numpy.zeros(len(date_list)) + a2_low,linewidth=1.0,linestyle="--",color="maroon")
ax1.plot(date_list,numpy.zeros(len(date_list)) + a2_low2,linewidth=0.5,linestyle=":",color="maroon")

ax1.plot(date_list,T_0_mean,linewidth=1.5,color="tomato",label="$WL_{2}(t)$")
ax1.plot(date_list,T_0_high2,linewidth=0.5,linestyle=":",color="tomato")
ax1.plot(date_list,T_0_high,linewidth=1.0,linestyle="--",color="tomato")
ax1.plot(date_list,T_0_low,linewidth=1.0,linestyle="--",color="tomato")
ax1.plot(date_list,T_0_low2,linewidth=0.5,linestyle=":",color="tomato")



# Adjust subplot
ax1.set_xlim(date_list[0],date_list[-1])
ax1.grid()
ax1.legend(loc=3, ncol=2)
ax1.set_xticklabels([])
ax1.set_ylabel("Water level (m rel to MLLW)")

at = AnchoredText(str('A'), prop=dict(size=10,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

# ---------------------------- Wave characteristics subplot ----------------------------

ax1 = plt.subplot(512)

q_95 = numpy.quantile(wave_output[:,1],.975)

# Plot wave characteristics
ax1.plot(date_list,wave_output[w_time,1],linewidth=1.,color="k",label="$F_{SWH}(t)$")
ax1.plot(date_list,wave_output[w_time,2],linewidth=1.,color="forestgreen",label="$F_{CS}(t)$")
ax1.plot(date_list,wave_output[w_time,3],linewidth=1.,color="mediumslateblue",label="$F_{AS}(t)$")
label1 = "97.5%ile\n" + str(numpy.round(q_95,1)) + " kJ"
ax1.plot([date_list[0],date_list[-1]],[q_95,q_95],linewidth=1.,color="red",label=label1)

# Adjust subplot
ax1.set_xlim(date_list[0],date_list[-1])
ax1.grid()
ax1.legend(loc=1)
ax1.set_xticklabels([])
ax1.set_ylabel("NOAA buoy 440915 wave energy (kJ)")

at = AnchoredText(str('B'), prop=dict(size=10,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

# -------------------------- Partial model contribution (f_wave/f_tide) subplot --------------------------

ax1 = plt.subplot(513)

# Calculate mean, quantiles for partial model contribution
A4_mean = numpy.mean(output_matrix_A4[burn_in:,:],axis=0)
A4_high2 = numpy.quantile(output_matrix_A4[burn_in:,:],0.975,axis=0)
A4_low2 = numpy.quantile(output_matrix_A4[burn_in:,:],0.025,axis=0)
A4_high = numpy.quantile(output_matrix_A4[burn_in:,:],0.16,axis=0)
A4_low = numpy.quantile(output_matrix_A4[burn_in:,:],0.84,axis=0)

A6_mean = numpy.mean(output_matrix_A6[burn_in:,:],axis=0)
A6_high2 = numpy.quantile(output_matrix_A6[burn_in:,:],0.975,axis=0)
A6_low2 = numpy.quantile(output_matrix_A6[burn_in:,:],0.025,axis=0)
A6_high = numpy.quantile(output_matrix_A6[burn_in:,:],0.16,axis=0)
A6_low = numpy.quantile(output_matrix_A6[burn_in:,:],0.84,axis=0)

# Plot partial model contribution (f_wave/f_tide)
ax1.plot(date_list,A4_mean ,linewidth=1.5,color="red"   ,label="$f_{tide}(t)$")
ax1.plot(date_list,A4_high2,linewidth=0.5,linestyle=":" ,color="red")
ax1.plot(date_list,A4_high ,linewidth=1.0,linestyle="--",color="red")
ax1.plot(date_list,A4_low  ,linewidth=1.0,linestyle="--",color="red")
ax1.plot(date_list,A4_low2 ,linewidth=0.5,linestyle=":" ,color="red")

ax1.plot(date_list,A6_mean ,linewidth=1.5,color="blue"  ,label="$f_{wave}(t)$")
ax1.plot(date_list,A6_high2,linewidth=0.5,linestyle=":" ,color="blue")
ax1.plot(date_list,A6_high ,linewidth=1.0,linestyle="--",color="blue")
ax1.plot(date_list,A6_low  ,linewidth=1.0,linestyle="--",color="blue")
ax1.plot(date_list,A6_low2 ,linewidth=0.5,linestyle=":" ,color="blue")

# Adjust subplot
ax1.set_xlim(date_list[0],date_list[-1])
ax1.grid()
ax1.legend(loc=3)
ax1.set_xticklabels([])
ax1.set_ylabel("Partial model component\ncontribution ($m^3$/m)")



at = AnchoredText(str('C'), prop=dict(size=10,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)


# ------------------------------ Modeled geomorphological response subplot ------------------------------

ax2 = plt.subplot(514)

# Calculate mean, quantiles of modeled volumetric change
semiempirical_mean = numpy.mean(output_matrix_A2[burn_in:,:],axis=0)
semiempirical_high2 = numpy.quantile(output_matrix_A2[burn_in:,:],0.975,axis=0)
semiempirical_low2 = numpy.quantile(output_matrix_A2[burn_in:,:],0.025,axis=0)
semiempirical_high = numpy.quantile(output_matrix_A2[burn_in:,:],0.16,axis=0)
semiempirical_low = numpy.quantile(output_matrix_A2[burn_in:,:],0.84,axis=0)

label2 = "$\int_{t_0}^{t} a_{1}(WL(t) - a_{2}) dt$ + " + "$\int_{t_0}^{t} a_{3}(WL(t) - WL_{0}(t)) dt$"
label_2_5 = ";\n $dWL_{0}/dt$"

a1_mean = numpy.mean(output_matrix_A[burn_in:,1],axis=0)
a1_std = numpy.std(output_matrix_A[burn_in:,1],axis=0)

a2_mean = numpy.mean(output_matrix_A[burn_in:,0],axis=0)
a2_std = numpy.std(output_matrix_A[burn_in:,0],axis=0)

a_mean = numpy.mean(output_matrix_A[burn_in:,3],axis=0)
a_std = numpy.std(output_matrix_A[burn_in:,3],axis=0)

k_mean = numpy.mean(output_matrix_A[burn_in:,4],axis=0)
k_std = numpy.std(output_matrix_A[burn_in:,4],axis=0)

label_2_65 = " = $\kappa^{-1}[WL(t) - WL_{0}(t)]$\n"
label_2_75 = "$a_{1}$ = " + str(round(a1_mean,1)) +" +/- " + str(round(a1_std,1)) + ", $a_{2}$ = " + str(round(a2_mean,1)) +" +/- " + str(round(a2_std,1))+",\n"
label_2_85 = "$a_{3}$ = " + str(round(a_mean,1)) +" +/- " + str(round(a_std,1)) + ", and $\kappa$ = " + str(round(k_mean,1)) +" +/- " + str(round(k_std,1)) + " h"

# Plot modeled volumetric change
ax2.plot(date_list,semiempirical_mean,linewidth=1.5,color="gray",label="$f_{geo}(t)$"		)
ax2.plot(date_list,semiempirical_high2,linewidth=0.5,linestyle=":",color="gray",label="95% CI")
ax2.plot(date_list,semiempirical_high,linewidth=1.0,linestyle="--",color="gray")
ax2.plot(date_list,semiempirical_low,linewidth=1.0,linestyle="--",color="gray")
ax2.plot(date_list,semiempirical_low2,linewidth=0.5,linestyle=":",color="gray")
ax2.plot(date_list[w_output],training_data,linewidth=0.0,marker="o",color="k",label="Surveys")

# Adjust subplot
ax2.legend(loc=1,fontsize=10)
ax2.set_ylim(-50,20)
ax2.grid()
ax2.set_xlim(date_list[0],date_list[-1])
ax2.set_xticklabels([])
ax2.set_ylabel("Volume change (m$^3$/m of shoreline)")

at = AnchoredText(str('D'), prop=dict(size=10,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax2.add_artist(at)



ax1 = plt.subplot(515)

ax1.set_ylim(-50,20)
ax1.set_xlim(date_list[0],date_list[-1])
ax1.set_xlabel("Date")
ax1.set_ylabel("Volume change (m$^3$/m of shoreline)")

at = AnchoredText(str('E'), prop=dict(size=10,fontweight='bold'), frameon=True, loc=4)
at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
ax1.add_artist(at)

plt.tight_layout()

pltname = wkspc + "SemiEmpirical_Holgate.png"
plt.savefig(pltname, dpi = 300)

date_list_file = wkspc + 'date_list.csv'
with open(date_list_file, 'r') as f:
    date_list_str = [line.strip() for line in f.readlines()]

date_list1 = numpy.array([datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in date_list_str])

# Read volumes
file_out = vol_wkspc + 'LBT_volumes.txt'
volumes_tmp = numpy.genfromtxt(file_out, delimiter=',')
volumes = volumes_tmp.copy()

gpsdata1 = glob.glob(vol_wkspc + "00_Transects/" + "*.csv")

output_LOO_CV = numpy.zeros((len(gpsdata1),7))

for idx_n1 in range(1,len(gpsdata1)):

	# Establish survey dates
	gpsdata1 = glob.glob(vol_wkspc + "00_Transects/" + "*.csv")
	survey_volumes = numpy.zeros((len(gpsdata1), 5))

	date_list_vols = []
	w_output = numpy.zeros(len(gpsdata1), dtype="int")

	for n, gpsdata in enumerate(gpsdata1):
		time_tmp_orig = gpsdata.split("_")[-1][0:-4]
		year_tmp, month_tmp, day_tmp = int(time_tmp_orig[0:4]), int(time_tmp_orig[4:6]), int(time_tmp_orig[6:])
		
		time_datum = datetime.datetime(year_tmp, month_tmp, day_tmp)
		survey_volumes[n, 0] = (time_datum - time_datum_start).total_seconds() / (3600.0*24.0)
		
		survey_volumes[n, 1:] = [numpy.mean(volumes[:, n])/25., numpy.mean(volumes[0:25, n])/25.,
								 numpy.mean(volumes[25:50, n])/25., numpy.mean(volumes[50:, n])/25.]
		
		date_list_vols.append([time_datum,])
		w_output[n] = numpy.where(tide_output[w_time, 0] == survey_volumes[n, 0])[0]

	survey_volumes[:, 1] = numpy.cumsum(survey_volumes[:, 1])
	mean_error = 5.65
	print("Error:", mean_error)

	# Setup parameters
	theta_guess = numpy.array ([-0.01,     numpy.mean(tide_output[:,1]), -0.01, 24.*14.,-0.01, 24.*14., -0.01, 24.*14., -0.00, 24.*14.,mean_error])
	theta_priors = numpy.array([3.5, numpy.mean(tide_output[:, 1])/10., 3.5, 24.*14., 0.5, 24.*14., 0.5, 24.*14., 0.5, 24.*14., mean_error])
	stepsizes = numpy.array([0.1, 0.1*.3048, 1., 50., 1., 50., 1., 50., 1., 50., 0.0]) /3.75

	# MCMC
	training_data = survey_volumes[:, 1].copy()
	print("...")
	print(idx_n1)
	print(len(training_data))
	index_1 = w_output[idx_n1]
	value = training_data[idx_n1]
	date = date_list1[index_1]
	
	training_data = numpy.delete(training_data,idx_n1)
	w_output = numpy.delete(w_output,idx_n1)

	# Function for MCMC calculation
	def function(theta):

		theta1 = theta[0:2]
		theta2 = theta[2:4]
		theta3 = theta[4:6]
		theta4 = theta[6:8]
		theta5 = theta[8:10]
		
		semi_empirical1 = semi_empirical_1(theta1,tide_output[:,1],dt)
		semi_empirical2 = semi_empirical_2(theta2,tide_output[:,1],numpy.mean(tide_output[:,1]),dt)
		semi_empirical3 = semi_empirical_2(theta3,wave_output[:,1],numpy.mean(wave_output[:,1]),dt)
		semi_empirical4 = semi_empirical_2(theta4,wave_output[:,2],numpy.mean(wave_output[:,2]),dt)	
		semi_empirical5 = semi_empirical_2(theta5,wave_output[:,3],numpy.mean(wave_output[:,3]),dt)

		semi_empirical = semi_empirical1 + semi_empirical2[0] + semi_empirical3[0] + semi_empirical4[0] + semi_empirical5[0]
		
		return semi_empirical[w_output],semi_empirical,semi_empirical2[1],semi_empirical1+semi_empirical2[0],theta1[1],semi_empirical3[0]+semi_empirical4[0]+semi_empirical5[0],semi_empirical4,semi_empirical5    

	
	MCMC_iters, burn_in = 100000, 50000
	
	print(len(training_data))
	print("...")
	output_matrix_A,output_matrix_A2,output_matrix_A3,output_matrix_A4,output_matrix_A5,output_matrix_A6,loglik_output = MCMC_20230607.MCMC(function,training_data,theta_guess,theta_priors,stepsizes,MCMC_iters)

	semiempirical_mean = numpy.mean(output_matrix_A2[burn_in:,index_1],axis=0)
	semiempirical_high2 = numpy.quantile(output_matrix_A2[burn_in:,index_1],0.975,axis=0)
	semiempirical_low2 = numpy.quantile(output_matrix_A2[burn_in:,index_1],0.025,axis=0)
	semiempirical_low = numpy.quantile(output_matrix_A2[burn_in:,index_1],0.16,axis=0)
	semiempirical_high = numpy.quantile(output_matrix_A2[burn_in:,index_1],0.84,axis=0)
	
	output_LOO_CV[idx_n1,0] = index_1
	output_LOO_CV[idx_n1,1] = value
	output_LOO_CV[idx_n1,2] = semiempirical_low2
	output_LOO_CV[idx_n1,3] = semiempirical_low
	output_LOO_CV[idx_n1,4] = semiempirical_mean
	output_LOO_CV[idx_n1,5] = semiempirical_high
	output_LOO_CV[idx_n1,6] = semiempirical_high2
	
	# Save results
	filename_base = wkspc + '03_Model_output/'
	numpy.savetxt(filename_base + "LOO_CV.txt", output_LOO_CV, delimiter=',')
	
# ----------------------------- Plot results -----------------------------

# ---------------------------- Water level subplot ----------------------------

date_list = date_list1[w_time]

# Plot modeled volumetric change

for n_plot in range(1,idx_n1+1):

	ax1.plot([date_list[int(output_LOO_CV[n_plot,0]-10.)],date_list[int(output_LOO_CV[n_plot,0]-10.)]],[output_LOO_CV[n_plot,3],output_LOO_CV[n_plot,5]],linewidth=1.5,color="gray")
	ax1.plot([date_list[int(output_LOO_CV[n_plot,0]-10.)],date_list[int(output_LOO_CV[n_plot,0]-10.)]],[output_LOO_CV[n_plot,2],output_LOO_CV[n_plot,6]],linewidth=0.0,marker="_",color="gray")
	ax1.plot([date_list[int(output_LOO_CV[n_plot,0]+10.)],date_list[int(output_LOO_CV[n_plot,0]+10.)]],[output_LOO_CV[n_plot,1]-5.65,output_LOO_CV[n_plot,1]+5.65],linewidth=1.0,color="k")
	ax1.plot([date_list[int(output_LOO_CV[n_plot,0]+10.)],date_list[int(output_LOO_CV[n_plot,0]+10.)]],[output_LOO_CV[n_plot,1]-2.*5.65,output_LOO_CV[n_plot,1]+2.*5.65],linewidth=0.0,marker="_",color="k")
	if n_plot == 1:

		ax1.plot(date_list[int(output_LOO_CV[n_plot,0]-10.)],output_LOO_CV[n_plot,4],linewidth=0.0,marker="o",color="gray",label="$f_{geo}(t)$")
		ax1.plot(date_list[int(output_LOO_CV[n_plot,0]+10.)],output_LOO_CV[n_plot,1],linewidth=0.0,marker="o",color="k",label="Surveys")
		
		
	if n_plot > 1:
		ax1.plot(date_list[int(output_LOO_CV[n_plot,0]-10.)],output_LOO_CV[n_plot,4],linewidth=0.0,marker="o",color="gray")
		ax1.plot(date_list[int(output_LOO_CV[n_plot,0]+10.)],output_LOO_CV[n_plot,1],linewidth=0.0,marker="o",color="k")
			
# Adjust subplot
ax1.legend(loc=1,fontsize=10)
ax1.grid()


# Save figure
plt.tight_layout()
pltname = wkspc + "SemiEmpirical_Holgate_LOOCV.png"
plt.savefig(pltname, dpi = 300)