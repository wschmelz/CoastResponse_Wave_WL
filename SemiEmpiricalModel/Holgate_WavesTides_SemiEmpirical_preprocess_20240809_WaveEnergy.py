import os
import glob
import numpy
import pandas
import datetime
import time
import loess2D

backslash = '\\'

wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

print (str(__file__))
print (time.strftime("%H:%M:%S"))
print ()
# ----------------------------- Define functions to load data -----------------------------

def load_tide_data(workspace):

	tide_datafiles = glob.glob(workspace + "/*.csv")
	count = 0 
	for file_out in tide_datafiles:
		df = pandas.read_csv(file_out,skiprows=1, header=None)

		tide_data_tmp1 = numpy.array(df)

		x1 = pandas.to_datetime(tide_data_tmp1[:,0])
		x2 = pandas.to_datetime(tide_data_tmp1[:,1])

		tide_data_tmp = numpy.zeros((len(tide_data_tmp1[:,0]),5))
		
		tide_data_tmp[:,0] = x1.year
		tide_data_tmp[:,1] = x1.month
		tide_data_tmp[:,2] = x1.day
		tide_data_tmp[:,3] = x2.hour
		tide_data_tmp[:,4] = tide_data_tmp1[:,4]

		if count ==0:
			tide_data = tide_data_tmp
		else:
			tide_data = numpy.append(tide_data,tide_data_tmp,axis=0)
		
		count+=1
			
	return tide_data
	
def load_wave_data(workspace):

	wave_datafiles = glob.glob(workspace + "/*.txt")
	count = 0   
	for file_out in wave_datafiles:

		wave_data_tmp = numpy.genfromtxt(file_out,skip_header=2)

		if count == 0:
			wave_data = wave_data_tmp * 1.0
		else:
			wave_data = numpy.append(wave_data,wave_data_tmp,axis=0)	
			
		count+=1
	return wave_data
	
# ----------------------------- Define functions to process data -----------------------------

def process_tide(tide_data):
	
	t1 = time.time()
		
	tide_output_tmp = numpy.zeros((len(tide_data[:,0]),2))

	date_list = []

	for n2 in range(0,len(tide_data[:,0])):	

		year_tmp = int(tide_data[n2,0])
		month_tmp = int(tide_data[n2,1])
		day_tmp = int(tide_data[n2,2]) 
		hour_tmp = int(tide_data[n2,3])

		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)

		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
		time_tmp_years = (time_datum_stop - time_datum_start).total_seconds() / (86400*365.25)

		date_list.append([time_datum_stop,])

		tide_output_tmp[n2,0] = time_tmp
		tide_output_tmp[n2,1] = tide_data[n2,4] * 0.3048   

	w1 = numpy.where((tide_output_tmp[:,0]>=time_tmp_first)&(tide_output_tmp[:,0]<time_tmp_last))[0]

	tide_output = tide_output_tmp[w1,:]
	date_list = numpy.array(date_list)[w1]

	w2 = numpy.where(numpy.isnan(tide_output[:,1])==False)[0]
	t2 = time.time()

	print("Minutes to process tide data:",(t2-t1)/60.)
	print("All n, tide data:",len(tide_output[:,1]))
	print("Valid n, tide data:",len(w2))
	print("Percent n valid:",round((float(len(w2))/len(tide_output[:,1]))*100.,2))
	print()
	
	return tide_output, tide_output_tmp, date_list	

	
def process_wave(wave_data,tide_output_tmp):
	
	t1 = time.time()

	wave_data_tmp = wave_data.copy()

	w1 = numpy.where((wave_data_tmp[:,8]<90.)&(wave_data_tmp[:,11]<400.))[0]

	wave_data = wave_data_tmp[w1,:] * 1.0
	wave_output_tmp = numpy.zeros((len(wave_data[:,0]),5))

	w1 = numpy.where((wave_data[:,11]<90.)&(wave_data[:,11]>=0.))[0]
	w2 = numpy.where((wave_data[:,11]<180.)&(wave_data[:,11]>=90.))[0]
	w3 = numpy.where((wave_data[:,11]<270.)&(wave_data[:,11]>=180.))[0]
	w4 = numpy.where((wave_data[:,11]<360.)&(wave_data[:,11]>=270.))[0]

	wave_output_tmp[w1,1] = (wave_data[w1,11] * -1.) + 90.
	wave_output_tmp[w2,1] = (wave_data[w2,11] * -1.) + 450.
	wave_output_tmp[w3,1] = (wave_data[w3,11] * -1.) + 450.
	wave_output_tmp[w4,1] = (wave_data[w4,11] * -1.) + 450.

	baseline_ang = 58.21

	wave_output_ang_tmp = wave_output_tmp[:,1] + (90. - baseline_ang)

	w1 = numpy.where(wave_output_ang_tmp>360)[0]
	wave_output_ang_tmp[w1] = wave_output_ang_tmp[w1] - 360

	w2 = numpy.where((wave_output_ang_tmp<270.)&(wave_output_ang_tmp>90.))[0]

	wave_data[w2,8] = wave_data[w2,8]*0.0

	wave_output_tmp_2 = numpy.zeros((len(tide_output_tmp[:,0]),6)) * numpy.nan

	for n2 in range(0,len(wave_data[:,0])):	
		year_tmp = int(wave_data[n2,0])
		month_tmp = int(wave_data[n2,1])
		day_tmp = int(wave_data[n2,2])
		hour_tmp = int(wave_data[n2,3])
		min_tmp = int(wave_data[n2,4])

		time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp, hour_tmp)

		time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)
		w_tidematch = numpy.where(tide_output_tmp[:,0]==time_tmp)[0]
		g_h = (1029. * 9.8)/8.
		if len(w_tidematch) > 0:
			
			wave_output_tmp_2[w_tidematch,0] = time_tmp
			wave_output_tmp_2[w_tidematch,1] = wave_output_tmp[n2,1]
			wave_output_tmp_2[w_tidematch,2] = ((((wave_data[n2,8] ** 2.)*g_h))/1000.)
			wave_output_tmp_2[w_tidematch,3] = ((((wave_data[n2,8] ** 2.)*g_h))/1000.) * numpy.cos(numpy.radians(wave_output_ang_tmp[n2]))
			wave_output_tmp_2[w_tidematch,4] = ((((wave_data[n2,8] ** 2.)*g_h))/1000.) * numpy.sin(numpy.radians(wave_output_ang_tmp[n2]))
			wave_output_tmp_2[w_tidematch,5] = numpy.radians(wave_output_ang_tmp[n2])
			
	w2 = numpy.where(numpy.isnan(wave_output_tmp_2[:,1])==False)[0]

	wave_output_tmp_2[:,0] = tide_output_tmp[:,0]

	time_wave = wave_output_tmp_2[:,0]*1.0
	wave_output_tmp = numpy.zeros((len(wave_output_tmp_2[:,0]),4))
	wave_output_tmp[:,0] = time_wave

	wave_output_tmp[:,1] = loess2D.weighted_linear(time_wave,time_wave*0.0,wave_output_tmp_2[:,0],wave_output_tmp_2[:,0]*0.0,wave_output_tmp_2[:,2],12,0.5,0.5)[2]
	wave_output_tmp[:,2] = loess2D.weighted_linear(time_wave,time_wave*0.0,wave_output_tmp_2[:,0],wave_output_tmp_2[:,0]*0.0,wave_output_tmp_2[:,3],12,0.5,0.5)[2]
	wave_output_tmp[:,3] = loess2D.weighted_linear(time_wave,time_wave*0.0,wave_output_tmp_2[:,0],wave_output_tmp_2[:,0]*0.0,wave_output_tmp_2[:,4],12,0.5,0.5)[2]
	
	w2 = numpy.where(numpy.isnan(wave_output_tmp[:,1])==False)[0]
	print("All n, wave data:",len(wave_output_tmp[:,1]))
	print("Valid n, wave data:",len(w2))
	print("Percent n valid:",round((float(len(w2))/len(wave_output_tmp[:,1]))*100.,2))
	print()
	

	w1 = numpy.where(numpy.isnan(wave_output_tmp[:,1]))[0]

	wave_output_tmp[w1,2] = -999.
	zeros = len(numpy.where(wave_output_tmp[w1,2]<0)[0])
	while zeros>0:
	
		wave_output_tmp[w1,1] = numpy.nanmean(wave_output_tmp[:,1]) + numpy.random.normal(loc=0,scale=numpy.nanstd(wave_output_tmp[:,1]),size=len(w1))
		w4 = numpy.where(wave_output_tmp[w1,1]<=0.0)[0]
		wave_output_tmp[w1[w4],1] = 0.0
			
		mult_CS = numpy.random.normal(loc=numpy.nanmean(wave_output_tmp_2[:,5]),scale=numpy.nanstd(wave_output_tmp_2[:,5]),size=len(w1))
		mult_CS_degs = numpy.degrees(mult_CS)

		w6 = numpy.where(mult_CS_degs>360)[0]
		mult_CS[w6] = numpy.radians(numpy.degrees(mult_CS[w6]) - 360)
		
		w5 = numpy.where((mult_CS_degs<270.)&(mult_CS_degs>90.))[0]
		
		for n3 in range(0,len(w5)):

			while ((numpy.degrees(mult_CS[w5[n3]])<270.)&(numpy.degrees(mult_CS[w5[n3]])>90.)):
				mult_CS[w5[n3]] = numpy.random.normal(loc=numpy.nanmean(wave_output_tmp_2[:,5]),scale=numpy.nanstd(wave_output_tmp_2[:,5]),size=1)
				if numpy.degrees(mult_CS[w5[n3]]) > 360:
					mult_CS[w5[n3]] = numpy.radians(numpy.degrees(mult_CS[w5[n3]])-360)
					
		wave_output_tmp[w1,2] = wave_output_tmp[w1,1] * numpy.cos(mult_CS)
		wave_output_tmp[w1,3] = wave_output_tmp[w1,1] * numpy.sin(mult_CS) 
		
		w7 = numpy.where(((wave_output_tmp[w1,2])>wave_output_tmp[w1,1])|((wave_output_tmp[w1,3])>wave_output_tmp[w1,1]))[0]
		if len(w7)>0:
			for n4 in range(0,len(w7)):
				tmp_idx = w7[n4]
				while (((wave_output_tmp[w1[tmp_idx],2])>wave_output_tmp[w1[tmp_idx],1])|((wave_output_tmp[w1[tmp_idx],3])>wave_output_tmp[w1[tmp_idx],1])):
					
					wave_output_tmp[w1[tmp_idx],1] = numpy.nanmean(wave_output_tmp[:,1]) + numpy.random.normal(loc=0,scale=numpy.nanstd(wave_output_tmp[:,1]),size=len(w1[tmp_idx]))
					w4 = numpy.where(wave_output_tmp[w1[tmp_idx],1]<=0.0)[0]
					tmp_idx_2 = w1[tmp_idx]
					wave_output_tmp[tmp_idx_2[w4],1] = 0.0
						
					mult_CS = numpy.random.normal(loc=numpy.nanmean(wave_output_tmp_2[:,5]),scale=numpy.nanstd(wave_output_tmp_2[:,5]),size=len(tmp_idx))
					mult_CS_degs = numpy.degrees(mult_CS)

					w6 = numpy.where(mult_CS_degs>360)[0]
					mult_CS[w6] = numpy.radians(numpy.degrees(mult_CS[w6]) - 360)
					
					w5 = numpy.where((mult_CS_degs<270.)&(mult_CS_degs>90.))[0]
					
					for n3 in range(0,len(w5)):

						while ((numpy.degrees(mult_CS[w5[n3]])<270.)&(numpy.degrees(mult_CS[w5[n3]])>90.)):
							mult_CS[w5[n3]] = numpy.random.normal(loc=numpy.nanmean(wave_output_tmp_2[:,5]),scale=numpy.nanstd(wave_output_tmp_2[:,5]),size=1)
							if numpy.degrees(mult_CS[w5[n3]]) > 360:
								mult_CS[w5[n3]] = numpy.radians(numpy.degrees(mult_CS[w5[n3]])-360)
								
					wave_output_tmp[w1[tmp_idx],2] = wave_output_tmp[w1[tmp_idx],1] * numpy.cos(mult_CS)
				wave_output_tmp[w1[tmp_idx],3] = wave_output_tmp[w1[tmp_idx],1] * numpy.sin(mult_CS) 
				
		w1 = numpy.where(wave_output_tmp[:,2]<0)[0]
		zeros = len(w1)

	w1 = numpy.where((wave_output_tmp[:,0]>=time_tmp_first)&(wave_output_tmp[:,0]<time_tmp_last))[0]
	wave_output = wave_output_tmp[w1,:]
	
	date_list = []
	for n2 in range(0,len(wave_output_tmp[:,0])):	

		time_datum_stop = time_datum_start + datetime.timedelta(days = wave_output_tmp[n2,0])

		date_list.append([time_datum_stop,])

	date_list = numpy.array(date_list)[w1]
	
	w1 = numpy.where((wave_output[:,0]>=time_tmp_first)&(wave_output[:,0]<time_tmp_last))[0]
	
	wave_output = wave_output[w1,:]* 1.0
	date_list = numpy.array(date_list)[w1]
	
	w2 = numpy.where(numpy.isnan(wave_output[:,1])==False)[0]

	t2 = time.time()

	print("Minutes to process wave data:",(t2-t1)/60.)
	print("All n, wave data:",len(wave_output[:,1]))
	print("Valid n, wave data:",len(w2))
	print("Percent n valid:",round((float(len(w2))/len(wave_output[:,1]))*100.,2))
	print()
	
	return wave_output, date_list	

# ----------------------------- Define relevant time period -----------------------------

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

time_tmp_last = (time_datum_last_data - time_datum_start).total_seconds() / (3600.0*24.)
time_tmp_first = (time_datum_first_data - time_datum_start).total_seconds() / (3600.0*24.)

# ----------------------------- Process Tide Data -----------------------------

tide_dir = wkspc + "01_Station8534720/"

tide_datafiles = glob.glob(tide_dir + "*.csv")

tide_data = load_tide_data(tide_dir)

filename = wkspc+"Station8534720_Tides_2015_2022.txt"

numpy.savetxt(filename, tide_data, delimiter=',')

tide_output, tide_output_tmp, date_list_tide = process_tide(tide_data)

# ----------------------------- Process Wave Data -----------------------------

wave_dir = wkspc + "00_Buoy44091/"

wave_data = load_wave_data(wave_dir)

filename = wkspc +"Buoy44091_Waves_2015_2022.txt"
numpy.savetxt(filename, wave_data)

wave_output, date_list_wave = process_wave(wave_data,tide_output_tmp)

# ----------------------------- Save processed data -----------------------------

if len(numpy.where(date_list_tide==date_list_wave)[0]==len(date_list_wave)):
	
	tide_file = wkspc + 'tide_output.csv'
	numpy.savetxt(tide_file, tide_output, delimiter=',')
	wave_file = wkspc + 'wave_output.csv'
	numpy.savetxt(wave_file,wave_output, delimiter=',')

	
	date_list_tide_str = []
	for idx in date_list_tide:
		for dt in idx:
			date_list_tide_str.append(dt.strftime('%Y-%m-%d %H:%M:%S'))	
	
	date_list_file = wkspc + 'date_list.csv'
	numpy.savetxt(date_list_file, date_list_tide_str, fmt='%s', delimiter=',')
	print("Data processed")
	
else:
	print("Wave/Tide date/time mismatched")