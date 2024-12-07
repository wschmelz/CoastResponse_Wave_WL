import arcpy, os, glob, numpy


from arcpy.sa import *
from arcpy import env
import time
arcpy.CheckOutExtension("Spatial")
arcpy.CheckOutExtension("3D")
env.overwriteOutput = True
arcpy.overwriteOutput = True

backslash = '\\'

sr = str("PROJCS['NAD_1983_UTM_Zone_18N',GEOGCS['GCS_North_American_1983',DATUM['D_North_American_1983',SPHEROID['GRS_1980',6378137.0,298.257222101]],PRIMEM['Greenwich',0.0],UNIT['Degree',0.0174532925199433]],PROJECTION['Transverse_Mercator'],PARAMETER['False_Easting',500000.0],PARAMETER['False_Northing',0.0],PARAMETER['Central_Meridian',-75.0],PARAMETER['Scale_Factor',0.9996],PARAMETER['Latitude_Of_Origin',0.0],UNIT['Meter',1.0]]")

backslash = '\\'
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"
wkspc_data = wkspc + '00_Data/'
gpsdata = glob.glob(wkspc_data + "*/")
gpsdata1 = glob.glob(gpsdata[0] + "02_Transects/" + "*.csv")
time_tmp_orig = gpsdata1[0].split("_")[-1][0:-4]
year_tmp = int(time_tmp_orig[0:4])
month_tmp = int(time_tmp_orig[4:6])
day_tmp = int(time_tmp_orig[6:])
time_datum_start = datetime.datetime(year_tmp, month_tmp, day_tmp)
file_out = wkspc + 'processed_data_GPR.csv'    
topo_data_GPR = numpy.genfromtxt(file_out,delimiter=',')

for n in range(0,len(gpsdata)):
	print gpsdata[n]
	gpsdata1 = glob.glob(gpsdata[n] + "02_Transects/" + "*.csv")
	time_tmp_orig = gpsdata1[0].split("_")[-1][0:-4]
	year_tmp = int(time_tmp_orig[0:4])
	month_tmp = int(time_tmp_orig[4:6])
	day_tmp = int(time_tmp_orig[6:])
	time_datum_stop = datetime.datetime(year_tmp, month_tmp, day_tmp)
	time_tmp = (time_datum_stop - time_datum_start).total_seconds() / (3600.0*24.)    
	time_datum = datetime.datetime(year_tmp, month_tmp, day_tmp)
	w1 = numpy.where(topo_data_GPR[:,0]==time_tmp)[0]
	time_data_tmp = numpy.zeros((len(w1),3))
	time_data_tmp[:,0] = topo_data_GPR[w1,1]
	time_data_tmp[:,1] = topo_data_GPR[w1,2]
	time_data_tmp[:,2] = topo_data_GPR[w1,3]

	file_out = wkspc + 'LBT_3D_' + str(time_tmp_orig) + '.csv'    

	header = "X,Y,Z"
	numpy.savetxt(file_out, time_data_tmp, delimiter=',', header=header, comments="")	
	
	file_out = os.path.join(wkspc, 'LBT_3D_' + str(time_tmp_orig) + '.csv')
	file_out2 = wkspc + 'LBT_3D_' + str(time_tmp_orig) + '.shp'

	in_Table = 'LBT_3D_' + str(time_tmp_orig) + '.csv'
	x_coords = "X"
	y_coords = "Y"
	z_coords = "Z"
	out_Layer = "layer1"
		
	arcpy.MakeXYEventLayer_management(in_Table, x_coords, y_coords, out_Layer, sr, z_coords)

	shape_name = 'LBT_3D_' + str(time_tmp_orig)

	wkspc1 = gpsdata[n]
	saved_Shapefile = os.path.join(wkspc1, 'LBT_3D_' + str(time_tmp_orig) + '.shp')
	arcpy.CreateFeatureclass_management(out_path=wkspc1, out_name=shape_name, geometry_type='POINT', spatial_reference=sr)

	fields = ["X", "Y", "Z"]
	for field in fields:
		arcpy.AddField_management(saved_Shapefile, field, "FLOAT")

	arcpy.MakeXYEventLayer_management(in_Table, x_coords, y_coords, out_Layer, sr, z_coords)

	arcpy.Append_management([out_Layer], saved_Shapefile, 'NO_TEST')