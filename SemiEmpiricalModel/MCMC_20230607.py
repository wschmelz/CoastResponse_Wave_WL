import numpy
import os
import csv
import decimal
from decimal import Decimal
from numba import njit
import warnings 

warnings.filterwarnings("ignore")

con = decimal.getcontext()
con.prec = 100
con.Emin = -9999999999
con.Emax =  9999999999

backslash = "\\"
wkspc = str(os.getcwd()).replace(backslash,"/") + "/"

model_out_path = wkspc + "03_Model_output/"

try:
	os.mkdir(model_out_path)
except OSError as error:
	pass

def log_lik(y,model,stdev):

	n_in = len(y)
	loglik = Decimal('0.0')
	a1 = Decimal(- ((n_in/2.)))
	a2 = Decimal(stdev**2.)
	b1 = Decimal((n_in/2.))
	b2 = Decimal(2. * numpy.pi)
	c = Decimal(numpy.sum(((y-model)**2.)/(2.*(stdev**2))))
	loglik = (a1*a2.ln()) -  (b1 * b2.ln()) - c
	-1. * float(loglik)
	return -1. * float(loglik)
	
def log_lik_vec(y,model,stdev):

	n_in = 1.
	loglik = Decimal('0.0')
	for deci in range(0,len(model)):
		a1 = Decimal(- ((n_in/2.)))
		a2 = Decimal(stdev[deci]**2.)
		b1 = Decimal((n_in/2.))
		b2 = Decimal(2. * numpy.pi)
		c = Decimal(numpy.sum(((y[deci]-model[deci])**2.)/(2.*(stdev[deci]**2))))
		d = a1 * a2.ln()
		e = b1 * b2.ln()
		f = d-e
		g = f - c
		loglik = loglik + g
		
	loglik = -1.0 * float(loglik)	
	return loglik
def log_lik_float(y,model,stdev):

	n_in = 1.
	loglik = Decimal('0.0')

	a1 = Decimal(- ((n_in/2.)))
	a2 = Decimal(stdev**2.)
	b1 = Decimal((n_in/2.))
	b2 = Decimal(2. * numpy.pi)
	c = Decimal(numpy.sum(((y-model)**2.)/(2.*(stdev**2))))
	d = a1 * a2.ln()
	e = b1 * b2.ln()
	f = d-e
	g = f - c
	loglik = loglik + g
		
	loglik = -1.0 * float(loglik)	
	return loglik

def MCMC(function,training_data,theta_guess,theta_priors,stepsizes,MCMC_iters):

	file_out_1 = wkspc+"03_Model_output/" + "posterior_params_MCMC.csv"
	file_out_2 = wkspc+"03_Model_output/" + "loglik_output_MCMC.csv"
	function_out,function_out2,function_out3,function_out4,function_out5,function_out6,function_out7,function_out8 = function(theta_guess)
	output_matrix_A = numpy.zeros((MCMC_iters,len(theta_guess)))
	output_matrix_A2 = numpy.zeros((MCMC_iters,len(function_out2)))
	output_matrix_A3 = numpy.zeros((MCMC_iters,len(function_out3)))
	output_matrix_A4 = numpy.zeros((MCMC_iters,len(function_out4)))
	output_matrix_A5 = numpy.zeros((MCMC_iters,1))
	output_matrix_A6 = numpy.zeros((MCMC_iters,len(function_out6)))

	loglik_output = numpy.zeros((MCMC_iters,1))
	accept_output = numpy.zeros((MCMC_iters,1))
	iter_vec= numpy.arange(0,MCMC_iters,1)
	index_to_change = 0	
	
	for n in range(MCMC_iters):

		if n==0:

			old_theta = theta_guess * 1.0
			new_theta  = theta_guess * 1.0
			theta_guess_orig = theta_guess * 1.0
			
			function_out,function_out2,function_out3,function_out4,function_out5,function_out6,function_out7,function_out8 = function(new_theta)
			model_err = numpy.absolute(new_theta[-1] * 1.0)
			#print(log_lik_vec(new_theta,theta_guess_orig,theta_priors)  )
			old_loglik = log_lik(function_out,training_data,model_err) + log_lik_vec(new_theta,theta_guess_orig,theta_priors)

		if n > 0:
			old_theta  = output_matrix_A[n-1,:]
			old_loglik = loglik_output[n-1,0] 
			
			new_theta[0:int(len(new_theta)-1)] = numpy.random.normal(loc = old_theta[0:int(len(new_theta)-1)] , scale = stepsizes[0:int(len(new_theta)-1)] )
			#new_theta[index_to_change] = numpy.random.normal(loc = old_theta[index_to_change], scale = stepsizes[index_to_change])

			while (new_theta[0] > 0.):
				new_theta[0] = numpy.random.normal(loc = old_theta[0], scale = stepsizes[0])	
			'''
			while (new_theta[1] < 0.):
				new_theta[1] = numpy.random.normal(loc = old_theta[1], scale = stepsizes[1])
			'''
			while (new_theta[2] > 0.):
				new_theta[2] = numpy.random.normal(loc = old_theta[2], scale = stepsizes[2])					
			
			while (new_theta[3] < 0.):
				new_theta[3] = numpy.random.normal(loc = old_theta[3], scale = stepsizes[3])
			'''
			while (new_theta[4] > 0.):
				new_theta[4] = numpy.random.normal(loc = old_theta[4], scale = stepsizes[4])					
			'''
			while (new_theta[5] < 0.):
				new_theta[5] = numpy.random.normal(loc = old_theta[5], scale = stepsizes[5])
			'''
			while (new_theta[6] > 0.):
				new_theta[6] = numpy.random.normal(loc = old_theta[6], scale = stepsizes[6])					
			'''
			while (new_theta[7] < 0.):
				new_theta[7] = numpy.random.normal(loc = old_theta[7], scale = stepsizes[7])
			while (new_theta[9] < 0.):
				new_theta[9] = numpy.random.normal(loc = old_theta[9], scale = stepsizes[9])
			
			index_to_change = index_to_change + 1
			
			if index_to_change == len(new_theta)-1:
				index_to_change = 0	
		
		function_out,function_out2,function_out3,function_out4,function_out5,function_out6,function_out7,function_out8 = function(new_theta)
		model_err = numpy.absolute(new_theta[-1] * 1.0)

		new_loglik =log_lik(function_out,training_data,model_err) + log_lik_vec(new_theta,theta_guess_orig,theta_priors)
		#print(log_lik_vec(new_theta,theta_guess_orig,theta_priors)  )
		if numpy.isnan(new_loglik) == False:
			if (new_loglik < old_loglik):
				output_matrix_A[n,:]  = new_theta
				output_matrix_A2[n,:] = function_out2
				output_matrix_A3[n,:] = function_out3
				output_matrix_A4[n,:] = function_out4
				output_matrix_A5[n,:] = function_out5
				output_matrix_A6[n,:] = function_out6
				loglik_output[n,0] = new_loglik
				accept_output[n,0] = 1.0

			else:
				u = numpy.random.uniform(0.0,1.0)

				if (u < numpy.exp(old_loglik - new_loglik)):
					output_matrix_A[n,:]  = new_theta
					output_matrix_A2[n,:] = function_out2
					output_matrix_A3[n,:] = function_out3
					output_matrix_A4[n,:] = function_out4
					output_matrix_A5[n,:] = function_out5
					output_matrix_A6[n,:] = function_out6
				
					loglik_output[n,0] = new_loglik
					accept_output[n,0] = 1.0

				else:
					output_matrix_A[n,:]  = old_theta
					output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
					output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
					output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
					output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
					output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 
					loglik_output[n,0] = old_loglik
					accept_output[n,0] = 0.0

		else:
			output_matrix_A[n,:]  = old_theta
			output_matrix_A2[n,:] = output_matrix_A2[n-1,:] 
			output_matrix_A3[n,:] = output_matrix_A3[n-1,:] 
			output_matrix_A4[n,:] = output_matrix_A4[n-1,:] 
			output_matrix_A5[n,:] = output_matrix_A5[n-1,:] 
			output_matrix_A6[n,:] = output_matrix_A6[n-1,:] 

			loglik_output[n,0] = old_loglik
			accept_output[n,0] = 0.0
		
		if (n+1) % 5000 == 0:
			print("")
			print ("Iteration:",n+1)
			print ("Posterior probability:",loglik_output[n,0])
			print ("Accept rate:",numpy.mean(accept_output[0:n,0]))
			print ("Parameters:",output_matrix_A[n,:])

		if (n+1) % 20000 ==0:

			with open(file_out_1, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(output_matrix_A)					
			with open(file_out_2, 'w',newline="") as csvfile2:
				writer = csv.writer(csvfile2)
				writer.writerows(loglik_output)	
				
	with open(file_out_1, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(output_matrix_A)					
	with open(file_out_2, 'w',newline="") as csvfile2:
		writer = csv.writer(csvfile2)
		writer.writerows(loglik_output)	
		
	return output_matrix_A,output_matrix_A2,output_matrix_A3,output_matrix_A4,output_matrix_A5,output_matrix_A6,loglik_output