Scripts and data for 

Semi-empirically modeling barrier sediment transport in response to hydrodynamic forcing using UAV-derived topographical data (Holgate, New Jersey)


W. John Schmelz, Ashlyn Spector, Lauren Neitzke-Adamo, Kenneth G. Miller

Department of Earth and Planetary Sciences, Rutgers University, 610 Taylor Road, Piscataway, New Jersey 08854


Description:
This document outlines the scripts and processes involved in analyzing the topographical data collected from UAV-RTK surveys, estimating the beach and dune elevations, calculating volumetric changes, and applying a semi-empirical model.

Scripts to run to complete analysis given topographical data (provided in ...\GP_Regression\00_Data) and the wave (...\SemiEmpirical\00_Buoy44091) and tide (...\SemiEmpirical\01_Station8534720) data:

    GP_Regression (Compensate for gaps in the survey data):
        01_Nelder_mead_optimize_20230915.py: Parameter optimization using the Nelder-Mead method for the Gaussian Process Regression.
        02_GPR_varcompare_nm_20230915.py: Maps Gaussian process priors to sample locations.
        03_GPR_image_20230915.py: Produces images of the GP regression model results.
        04_Topo_extract.py: Creates .csv and .shp files that contain the original topographical data with gaps filled with the GP regression results.
		
    VolumeCalc (Calculate volumetric changes):
        3D_Analyst.py: This arcpy script calculates the volumetric changes utilizing the gap-filled UAV-derived DEMs. The .shp files produced by script 04_Topo_extract.py must be placed in the LBT_Holgate\03_Shapefiles directory.

    SemiEmpirical (Apply the semi-empirical model):
		Holgate_WavesTides_SemiEmpirical_preprocess_20230913.py: Processes the wave and tide data for use in the semi-empirical model.
        Holgate_WavesTides_SemiEmpirical_20230913.py: Relates the volumetric change information calculated using the 3D_Analyst.py script (i.e., the programmatic applicaiton of methods from Psuty et al., 2018, LBT_volumes.txt/Volumes_Pivot.xls) to the wave and tide data.
