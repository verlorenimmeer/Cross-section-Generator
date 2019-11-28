# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 11:14:56 2019

@author: dxa509
"""
import math
import scipy.integrate as integrate
import numpy as np
import random
import sympy
import matplotlib.pyplot as plot
from scipy import interpolate
from sympy import simplify
from ps2 import Annihilate,Particle, ParticleDatabase,DiracMatrices,ParticleData,FourVector
"""
The focus of this project is to produce a simulated particle event, in this case two-to-two processes using Monte Carlo. 
This is also used to calculate the cross-section integral. There are 4 main goals: transforming the two-to-two partonic cross-sections given 
diﬀerentially in ˆ t to diﬀerential cross-sections in cos ˆ θ and integrating them, using the accept-and-reject method to generate two-body 
ﬁnal state events, calculating the PDF for a given particle and calculating the total crosssection
"""
#Goal 1:  Transform the two-to-two partonic cross-sections given here diﬀerentially in ˆ t to diﬀerential crosssections in cos ˆ θ.
# Integrate these to ﬁnd the total partonic cross-section, given ˆ s.

class functions():
    """
    
    Class defining different functions for (d crosssection)/(d costheta).
    
    """
    
    def req_for_eq(self,s,cos_theta):
        """
        Method creating set of variables used in every function.
        __Innit__ was not used because cos_theta is not given only when class is called but for some methods in particular
        """
        self.alpha=0.1383                                                        #initialise alpha value; it can vary, took the value found in pythia
        self.s=s                                                                #initialise s; value given by user
        self.cost=cos_theta                                                     #initialise cos_theta; it will be given by randomly choosing it in the algorithm
        self.t=-s/2*(1-self.cost)                                               #calculate t as a function of cos theta
        self.u=-s/2*(1+self.cost)                                               #calculate u as a function of cos theta
        self.dt_dcost=s/2                                                      #calculate dt/dcostheta to apply in the chain rule
    
    def quark_scattering(self,s,cos_theta):
        """
        Function for derivative of cross section when quark scattering
        """
        self.req_for_eq(s,cos_theta)                                            #call function with variables defined
        dsigma_dt=math.pi*pow(self.alpha,2)*(pow(self.s,2)+pow(self.u,2))*(9*self.s*self.u-4*pow(self.t,2))/(9*pow(self.s,3)*pow(self.t,2)*self.u) # derivative of cross section
        conversion_to_barn=0.4                                                  #constant for barn conversion
        dsigma_dcost=dsigma_dt*self.dt_dcost*conversion_to_barn                 #unit conversion
        
        return dsigma_dcost                                           #return values
    
    def gluon_scattering(self,s,cos_theta):
        """
        Function for derivative of coss section when gluon scattering
        """
        self.req_for_eq(s,cos_theta)                                            #call function with variables defined
        dsigma_dt=math.pi*pow(self.alpha,2)*(pow(self.t,2)+pow(self.u,2))*(4*self.s*self.s-9*self.t*self.u)/(24*pow(self.s,4)*self.t*self.u) # derivative of cross section
        conversion_to_barn=0.4                                                  #constant for barn conversion
        dsigma_dcost=dsigma_dt*self.dt_dcost*conversion_to_barn                 #unit conversion
        return dsigma_dcost                                                     #return values
    
    def higgs_production(self,s,cos_theta):
        """
        Function for derivative of coss section when higgs production
        """
        self.req_for_eq(s,cos_theta)                                            #call function with variables defined
        #for Higgs production some other constants are used; they depend on the particles involved and the input energy; i chose the ones for the energy equivalent to the CM
        Bgg=8.57/100
        Bff=6.5/100
        gama=4.07*pow(10,-3)
        mH=125.09                                                               #mass of higgs at rest
        dsigma_dt=math.pi*Bgg*Bff*pow(gama,2)/(8*(pow(gama,2)*pow(mH,2)+pow((s-pow(mH,2)),2)))  # derivative of cross section
        conversion_to_barn=0.4                                                  #constant for barn conversion
        dsigma_dcost=dsigma_dt*self.dt_dcost*conversion_to_barn                 #unit conversion
        return dsigma_dcost                                                     #return value
        
    def max_min_value(self,f,s):
        """
        Method calculating maximum for a given function
        """
        x_axis=np.arange(-1,1,0.1)                                 #create x axis with step 0.01. Given that the equations are of order 2 in theta the error on the max would be e-4
        print(x_axis)
        y_axis=[f(s,x) for x in x_axis]
        #a=[f(s,x) for x in x_axis]
        #print(a)
        from numpy import inf
        for x in range(len(x_axis)):
            if(y_axis[x]==inf):
                y_axis[x]=0
        print(y_axis)
        max_x = max(y_axis)                                  #find max
        #min_x=min([f(s,x) for x in x_axis]) 
        return max_x
    def integrate(self,f,s):
        """
        Method integrating the given function
        """
        result = integrate.quad(lambda x: f(s,x), -1, 1)                         #integrate the function on 0,1
        return result[0]                                                           #the result is 2*integral because the actual interval is -1 1 for an even function
 
#Goal 2:   Write a class that accepts the transformed diﬀerential cross-sections of Goal 1 as an argument, and generates two-body ﬁnal state events according to this distribution, given ˆ s.
# Use the accept-and-reject method to generate the events.    
class event_gen():
    """
    Class generating events based on the derivative of cross section for given s
    """
    def __init__(self,s,f,N):
        """
        Initialise values to work with. It calls the functions() class and calculate integral and maximum for given dcrossection so it doesn't do it during iteration in hit and miss
        """
        self.N_fail = 0                                                         #number of fails initialised to 0
        self.N_succ = 0                                                         #number of successes initialised to 0
        self.N=N                                                        
        self.s=s                                                      
        self.f=f                                                                #function that gives the derivative of cross section
        self.function = functions()                                             #call class for the functions
        self.average_f = 0                                                      #initialise an average for the cross section;will be used when calculating the integral with Monte Carlo
        self.max_xs = self.function.max_min_value(self.f,self.s)                #initialise max of dcross section
        #self.min_xs = self.function.max_min_value(self.f,self.s)[1] 
        self.output1=[]
        self.output2=[]
    def hit_and_miss(self):
        """
        Method applying the hit and miss algorithm. It gives FourVectors for output momentum
        """
            
        cos_theta_max=-1                                                    #initialise a max and min for cos theta to find the sampling range;theoretically it should be 2, from -1 to 1 but given that we use random generators it might be a bit smaller
        cos_theta_min=1
        while (self.N_succ<self.N):
            cos_theta = random.uniform(-1.0,1.0)                                #generate cos theta
            theta=math.acos(cos_theta)
            if(cos_theta>=cos_theta_max):                                       #update min and max for cos theta
                cos_theta_max=cos_theta
            if(cos_theta<=cos_theta_min):
                cos_theta_min=cos_theta
            
            dxs = self.f(self.s,cos_theta)                                      #give value for of d cross section for the chosen coss theta
    
            random_dxs = random.uniform(0,self.max_xs)                #generate a random between 0 and the maximum cross section
            if(abs(random_dxs)>abs(dxs)):                                       #check if it bigger than the calculated derivative
                self.N_fail += 1                                                #if it is, then it fails and tries again
                continue
            
            else:
                phi = random.uniform(0,2*math.pi)                               #else it chooses a phi
                q=math.sqrt(self.s)/2                                            #q, the energy element of the momentum Fourvector is calculated as sqrt(s)/2
                st=math.sin(theta)                                              #sin theta
                ct=math.cos(theta)                                              #cos theta
                sf=math.sin(phi)                                                #sin phi
                cf=math.cos(phi)                                                #cos phi
                p3=FourVector(q,q*st*cf,q*st*sf,q*ct)                           #momentum FourVector for p3         
                p4=FourVector(q,-q*st*cf,-q*st*sf,-q*ct)                        # momentum ForuVector for p4
                self.output1.append(p3)
                self.output2.append(p4)
                self.average_f += dxs                                           #add to the avearge d cross section
                self.N_succ += 1                                                #increase success
        return self.output1,self.output2
    def print_output(self):
        """
        Method prints output vectors for generated events to file that contains in name: the function and the value of s
        In the file there is information regarding the Number of events generated and the probability.
        """
        self.hit_and_miss()
        if(self.N_succ>0):
            result=self.average_f/self.N_succ                                       #calculate integral of dcross section using monte carlo formula as in Problem2 :sum(f)/n*[a-b]
            with  open('output_generator_s='+str(self.s)+'_'+f.__name__+'.txt', 'w') as outputfile:
                outputfile.write('Probability of succes: '+str(self.N_succ/(self.N_fail+self.N_succ))+'\n')
                outputfile.write('Integrated cross section: '+str(result)+' for s='+str(self.s)+' and N='+str(self.N)+'\n')
                for i in range(len(self.output1)):
                    outputfile.write('Output Momentum Four Vector 1:' + str(self.output1[i])+'\n')
                    outputfile.write('Output Momentum Four Vector 2:' + str(self.output2[i])+'\n')                                       #print results to file
        else:
            print('No succesful event production')
    def generate(self):
        """
        Method to be called for generation and print
        """
        self.hit_and_miss()
        self.print_output()
        

#Goal 3: Create a class to read in the PDF data ﬁle pdf.dat and return the PDF for a given parton with a momentum fraction x and at an energy scale Q. 
#The format of each line in the ﬁle is x and Q, followed by the PDF for the partons in the following order (given by PID) -5 -4 -3 -2 -1 1 2 3 4 5 21. Lines beginning with # are comments.
# You may use an interpolating package like interp2d from scipy. 
class PDF():
    def __init__(self,x,q,pid,file="pdf.dat"):
        """
        Constructor that on intialisation assigns values for file name, given x,q and pid
        """
        self.x_val=x                                                            #x value
        self.q_val=q                                                            #energy value
        self.pid_val=str(pid)                                                   #get pid as string
        self.file = file                                                        #input file
        self.pid_values=['-5','-4','-3','-2','-1','1','2','3','4','5','21']      #pid of particles for which PDFs are known in input file

    def read_and_populate_with_content(self):
        """
        Method used to read from the file
        If its read, the content is saved in a list of list of lists as follows: for each pid there is a list containing lists of lists as equivalent of a matrix/grid M[x][q]
        x - a momentum fraction
        Q - an energy scale
        """
        npid=len(self.pid_values)
        x_index=0
        self.values=[[] for i in range(npid)]
        self.x = []
        self.q = []
        fileObject = open(self.file, "r")
        for line in fileObject:
            fields = line.split()
            if fields[0] != "#":                                                #ensure that the row starting with comments wont be taken in account
                    if(float(fields[0]) not in self.x):                         #check if x has been found before
                        self.x.append(float(fields[0]))                         #add value to list
                        x_index+=1                                              #increase index
                    if(x_index<2):                                              #if only one x has been found
                        if(float(fields[1]) not in self.q):                     #add elements to q if not there
                            self.q.append(float(fields[1]))                     
                          
                            for i in range(npid):                               #for each particle id
                                self.values[i].append([])                       #add a new list
                    q_idx=self.q.index(float(fields[1]))                        #get index of current q
                    for i in range(2,len(fields)):                              #add value to list of lists; it starts at 2 because field 0 and 1 contain x and q
                        self.values[i-2][q_idx].append(float(fields[i]))        #index is x and q, as in a matrix
        return self.values

    def get_pdf_matrix_based_on_pid(self, pid):
        """
            Method that based on a given pid, it returns a list of lists with given values, x - being the equivalent of a line and q - being the equivalent of a column index
        :param pid: a value from ['-5','-4','-3','-2','-1','1','2','3','4','5','21']
        :return: a list of lists with pdf values, with the general form: pdf[x][q] = pdf_value
        """
        self.read_and_populate_with_content()
        index=self.pid_values.index(pid)
        self.pdf_matrix = self.values[index]

        return self.pdf_matrix

    def interpolate_function(self):
        """
        Method interpolates function for given pid on x and q with constructed list of lists from file
        """
        pdf_matrix = self.get_pdf_matrix_based_on_pid(self.pid_val)
        f = interpolate.interp2d(self.x,self.q,pdf_matrix)
        z = f(self.x_val,self.q_val)
        return z
    
    def get(self):
        """
        Method implemented only to keep a relevant name for interpolation and a short one for returning the value to the user
        """
        return self.interpolate_function()


#Goal 4:Calculate the total cross-section, integrating over x1, x2, and cos ˆ θ, using the PDF class. 
#When the energy scale is independent of ˆ θ the problem can be factorised. First, select a ˆ s from the cross-section integrated over cos ˆ θ.
# This requires sampling both x1 and x2. Next, select a cos ˆ θ for the partonic cross-section at the selected ˆ s.

def gen_from_pdf(f,N,pid):
    """
    Function calculates integral of f(x1)*f(x2)*dxs/dcostheta as a triple integral factorised when Q is independent of cos theta
    It samples x1 and x2 then calculates an s hat. S hat is used then to calculate dxs/dcostheta. 
    f-input function used for differential cross section
    N-number of itterations for Monte carlo integral
    pid- particle id, input for f(x1,Q) to tell which value from the grid should be used
    """
    average_x1=0
    average_x2=0
    average_dxs=0
    s=512                                                                      #Q max
    x1dif=1
    x2dif=1
    dcos_theta=2
    with  open('PDFgen_events_'+f.__name__+'.txt', 'w') as outputfile:
        outputfile.write('N='+str(N)+'\n')

    for i in range(N):
        x1 = random.uniform(0,1)
        x2 = random.uniform(0,1)
        
        s_hat=x1*x2*s
        p3,p4=event_gen(s_hat,f,1).hit_and_miss()                                   #can generate one event for each s calculated

        with  open('PDFgen_events_'+f.__name__+'.txt', 'a') as outputfile:
            outputfile.write('s='+str(s_hat)+' x1='+str(x1)+' x2='+str(x2)+'\n')
            outputfile.write('Output Momentum Four Vector 1:' + str(p3[0])+'\n')
            outputfile.write('Output Momentum Four Vector 2:' + str(p4[0])+'\n')                                       #print results to file
        
        f_x1=PDF(x1,s_hat,pid,file="pdf.dat").get()
        f_x2=PDF(x2,s_hat,pid,file="pdf.dat").get()
        
        average_x1+=f_x1
        average_x2+=f_x2
       
        cos_theta=random.uniform(-1,1)
        average_dxs+=f(s_hat,cos_theta)
    integral_x1=average_x1/N*x1dif                #calculate integral for x2
    integral_x2=average_x2/N*x2dif                #calculate integral for x1
    
    #integral_xs= functions().integrate(f,s_hat)
    integral_xs=average_dxs/N*dcos_theta           #calculate integral for dcross section
    with  open('PDFgen_events_'+f.__name__+'.txt', 'a') as outputfile:
            outputfile.write('Total cross-section = '+str(integral_x1*integral_x2*integral_xs)+'\n')

    return integral_x1*integral_x2*integral_xs     #return total integral.
 
#Trials
f=functions().gluon_scattering    #example of defining name of function
a=PDF(1.00e-06,3,-4).get()        #example of getting an intermediate value of PDF
b=gen_from_pdf(f,1000,'-2')     #example of generation from PDF for 1000 points on PID -2
event_gen(520,f,10).generate()   #example of 10 events generation

        