# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 09:22:18 2019

@author: Sebastian
"""

import numpy as np
from numpy.random import RandomState
from collections import defaultdict, OrderedDict


class NSNR(object):
    """
    Class that generates all arrival/inter_arrival times
    Parameters:
        sim_end_time: Simulation end time
        reps: Number of replications
        scv: squared coefficient of variation of X_{2}
        rho: \rho_1-lag 
    Methods:
        set_intervals_from_data: Feeds the data
        generate_arrival_times: Generate a list of arrival times, given the replications
        generate_interarrival_times: Generate a list of interarrival times, given the replications
        get_rho_range: Method that returns the feasible range of rho
        start_arrival: Method to initialize generator of NSNP arrival times
        generate_interarrival: Method that generates NSNP arrival times 
        set_service: set the service time mean
        generate_service: generate an observation for service time, first a service time mean has to be set
        _NSNPSim_Main: Main internal function that process the arrival/interarrival times
        _my_inv_rate_func: Internal function that returns values of the inverse rate function
        _my_cum_rate_func: Internal function that returns values of the cumulative rate function
        _new_get_MMECO: Internal function that sets collectors and computes the implied third momment
        _start_rep: Internal function that starts each replication
        _make_arrival: Internal function that generates an arrival
    """
    def __init__(self,sim_end_time,reps,scv,rho):
        self.sim_end_time = sim_end_time
        self.reps = reps
        self.scv = scv
        self.rho = rho
        self.CanMakeMMECO = True     #No se si cambiarlo y después cambiarlo 
        self.MajPhType = 5           # 1: exponential, 2: h2b, 3: MECon, 4: MECO (3-moment) and 5:Markov-MECO
        self.NumNodes = 3
        self.WithDSPP = False        #No estoy seguro comprobar
        #self.GammaN = 14               # Using Gamma(n,1)
        #self.ExponMean = 1              #using expo(mean)
        self.data_collector = dict()        #An array that collects every replication as a dictionary
        self.show_m = False
        
    def update_params(self, new_scv, new_rho):
        self.scv = new_scv
        self.rho = new_rho
    
    def set_intervals_from_data(self,data):
        """Storage of the data and put it available for other methods"""
        self.data = data

    def generate_arrival_times(self, stream):
        """
        Generate the arrival times given the simulation parameters given by the user
        Parameters:
            Stream: seed and observation generator
        Return:
            Data Collector: A dictionary of all replications
        """
        self.OutputArrivalTimes = True
        self._NSNPSim_Main(stream)
        return self.data_collector
        
    def generate_interarrival_times(self, stream):
        """
        Generate the interarrival times given the simulation parameters given by the user
        Parameters:
            Stream: seed and observation generator
        Return:
            Data Collector: A dictionary of all replications
        """
        self.OutputArrivalTimes = False
        self._NSNPSim_Main(stream)
        return self.data_collector

    def _NSNPSim_Main(self,stream):
        """
        Specify Stationary Majorizing Process
        Parameters:
            Stream: seed and observation generator
        Return:
            Void
        """
        #Call GetMajPh
        self._new_get_MMECO()
        if self.CanMakeMMECO == False:
            print("Corresponding Markov-MECO is not possible, reset the object or change parameters")
            exit()
        for j in range(self.reps):
            #Run the simulation until the end event
            self.curr_time = 0.0  #initialize time
            self._start_rep(stream)
            while self.curr_time < self.sim_end_time:
                self._make_arrival(stream)
                if self.OutputArrivalTimes:
                    self.data_collector['rep%d'%j] = self.arr_storage
                else:    
                    self.data_collector['rep%d'%j] = self.interarr_storage

    def start_arrival(self):
        """
        Method to initialize enviroment to generate arrivals one by one.
        No Parameters
        Return: Void
        """
        self._new_get_MMECO()
        self.first_arrival = True
        self.curr_time = 0.0 #itialize

    def generate_interarrival(self, stream):
        """
        Generates a  Markov-MECO observation one by one
        Parameters: 
            A Stream seed
        Return: 
            Observation
        """
        #WhichBasePh = lcgrand(CompMixProb)
        if self.first_arrival :
            if (self.WithDSPP):
                if self.GammaN > 25:
                    self.XVar = (1 / self.GammaN) * stream._GammaSpecOne(self.GammaN)
                else:
                    self.XVar = (1 / self.GammaN) * stream._ErlangOneStep(self.GammaN, 1)
            else:
                self.XVar = 1
            self.LastS = self.XVar * self._my_cum_rate_func(self.sim_end_time)

            try:
                WhichBasePh = stream.rand()
            except AttributeError:
                WhichBasePh = stream.randomly.rand()
            R = 0
            while True:
                if WhichBasePh < self.SteadyCumVector[R]:
                    break
                R = R + 1
            CurPhase = R
            if CurPhase < self.MMECOOrder:
                #initially in chain 1--generate Erlang of K+1-R, with rate1
                FirstArrTime = stream._ErlangOneStep(self.MMECOOrder - CurPhase, 1 / self.MMECORate1)
                self.PrevGenErlang = 1
            else:
                #initially in chain 2--generate Erlang of 2*K+1-R, with rate2
                FirstArrTime = stream._ErlangOneStep(2 * self.MMECOOrder - CurPhase, 1 / self.MMECORate2)
                self.PrevGenErlang = 2
            if FirstArrTime > self.LastS:
                InvFirstTime = self.sim_end_time + 1
            else:
                InvFirstTime = self._my_inv_rate_func(FirstArrTime/self.XVar)
                self.interarr_storage.append(InvFirstTime)
                self.arr_storage.append(InvFirstTime)
            self.first_arrival = False
            arrival_time = InvFirstTime - self.curr_time
            self.curr_time = InvFirstTime
            self.InvTime_ant = InvFirstTime
            return arrival_time
        else:
            try:
                WhichBasePh = stream.rand()
            except AttributeError:
                WhichBasePh = stream.randomly.rand()
            if self.PrevGenErlang == 1:
                if WhichBasePh < 1 - self.MMECOAlpha12:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate1)
                    self.PrevGenErlang = 1
                else:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate2)
                    self.PrevGenErlang = 2
            else:
                if WhichBasePh < self.MMECOAlpha21:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate1)
                    self.PrevGenErlang = 1
                else:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate2)
                    self.PrevGenErlang = 2
            # Calculate NS arrival time: R^-1(NextArrTime)
            self.curr_time = self.XVar * self._my_cum_rate_func(self.curr_time) #sim_time por self.curr_time
            if self.curr_time + NextArrTime > self.LastS:
                InvTime = self.sim_end_time + 1
            else:
                InvTime = self._my_inv_rate_func((self.curr_time + NextArrTime) / self.XVar)
                self.interarr_storage.append(InvTime-self.arr_storage[-1])
                self.arr_storage.append(InvTime)
            #print("InvTime",InvTime,"self.curr_time", self.curr_time)
            arrival_time = InvTime - self.InvTime_ant
            self.curr_time = InvTime
            self.InvTime_ant = InvTime
            return arrival_time        
 
    def set_service(self,service_mean):
        self.service_mean = service_mean

    def generate_service(self,stream):
        return stream._serv(self.service_mean)

    def _my_inv_rate_func(self,curr_value):
        """
        Generate the inverse rate function for a piecewise constant function (no equaly endpoints) in a given point
        Parameters:
            cur_value: Current value to convert
        Retunr:
            Observation
        """
        try:
            NumInts = len(self.data)
        except UnboundLocalError:
            print("The User has not yet fed the class with data")
        #IntVals =  np.empty(NumInts) #def 0 -> NumInts
        IntVals = self.data[:,1]
        
        IntEndPts = np.empty(NumInts+1) #def 0 -> NumInts+1
        IntEndPts[0] = 0.0 
        IntEndPts[1:-1] = self.data[1:,0]
        IntEndPts[-1] = self.sim_end_time
        
        FuncVals =  np.empty(NumInts+1) #def 0 -> NumInts+1
        FuncVals[0] = 0.0
        #Calculating the cummulative function rate R(t)
        for i in range(1,NumInts+1):
            FuncVals[i] = FuncVals[i-1] + IntVals[i-1]*(IntEndPts[i] - IntEndPts[i-1])
        # Determine which interval, get R^(-1)(t)
        i = 0
        while curr_value > FuncVals[i]:
            i += 1  
        return IntEndPts[i - 1] + (curr_value - FuncVals[i - 1]) / IntVals[i-1]

    def _my_cum_rate_func(self,curr_time):
        """
        This function does the acual heavy work -> Convert the function and transform from time to vals
        Parameter:
            curr_time: Current time of simulation
        Return:
            the cumulate rate function until current time of simulation
        """
        EndTime = self.sim_end_time
        try:
            NumInts = len(self.data)
        except UnboundLocalError:
            print("The User has not yet fed the class with data")
        IntVals = self.data[:,1]
        
        IntEndPts = np.empty(NumInts+1) #def 0 -> NumInts+1
        IntEndPts[0] = 0.0 
        IntEndPts[1:-1] = self.data[1:,0]
        IntEndPts[-1] = self.sim_end_time
        
        FuncVals =  np.empty(NumInts+1) #def 0 -> NumInts+1
        FuncVals[0] = 0.0
        #Calculating the cummulative function rate R(t)
        for i in range(1,NumInts+1):
            FuncVals[i] = FuncVals[i-1] + IntVals[i-1]*(IntEndPts[i] - IntEndPts[i-1])
        #(IGNORE: determine which interval, get R^-1(y))
        IntPt = 0
        while curr_time > IntEndPts[IntPt]:
            IntPt += 1
        # Assign current value of rate function
        return FuncVals[IntPt - 1] + IntVals[IntPt-1] * (curr_time - IntEndPts[IntPt - 1])
    
    def _new_get_MMECO(self):
        """
        Calculates the implied thir moment given the user's parameters
        Parameter:
            None
        Return:
            Void 
        """
        self.interarr_storage = []
        self.arr_storage = []
        ImpThirdMoms = 0 #inicializar implied third momemnt   
        MeanSvTime = 1
        TrueVarMoms = self.scv * np.power(MeanSvTime, 2)
        TrueSecMoms = TrueVarMoms + np.power(MeanSvTime, 2)
        Coeffvar = np.sqrt(self.scv)
        """Get Implied Third Moment"""
        if self.scv == 1:
            #Exponential
            TwoMomsOrder = 1
            ImpThirdMoms = 6 # E(X^3) = (1/\lambda) 3!, with \lambda = 1 
        elif self.scv < 1:
            #MECon: Mixture of Erlangs of Common order
            #Determine K
            k = 1
            while ((1/k) > self.scv):
                k += 1
            #Parameters
            self.MEConOrder = k
            self.MEConAlpha = (1 / (1 + self.scv) * (k * self.scv - np.sqrt(k*(1+self.scv)- (np.power(k,2)*self.scv)) ) )
            self.MEConRate = (k-self.MEConAlpha) / MeanSvTime
            # Steady State Vector
            self.SteadyVector = np.empty(2*k - 1)
            self.SteadyCumVector = np.empty(2*k - 1)
            for i in range(k):
                self.SteadyVector[i] = self.MEConAlpha / (k + self.MEConAlpha)
            for i in range(k,2*k-1):
                self.SteadyVector[i] = (1 - self.MEConAlpha)/(k + self.MEConAlpha)
            TwoMomsOrder = 2 * self.MEConOrder - 1
            TMat = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatSqInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatCuInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            ZetaVec = np.zeros(TwoMomsOrder)
            ZetaVec[0] =  1 - self.MEConAlpha
            ZetaVec[self.MEConOrder] = self.MEConAlpha
            #TMatInv directly. relative easy
            for i in range(self.MEConOrder):
                for j in range(i,self.MEConOrder):
                    TMatInv[i,j] = -1/self.MEConRate
            for i in range(self.MEConOrder,2*self.MEConOrder-1):
                for j in range(i,2*self.MEConOrder-1):
                    TMatInv[i,j] = -1 / self.MEConRate
        else:
            #H2B Hyperexponential
            TwoMomsOrder = 2
            TMat = np.zeros((TwoMomsOrder,TwoMomsOrder)) #This is D0 \equiv U(A1-I)
            TMatInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatSqInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatCuInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            ZetaVec = np.zeros(TwoMomsOrder) #This is zeta vec
            
            #get H2B Params
            self.SteadyVector = np.empty(TwoMomsOrder)
            self.SteadyCumVector = np.empty(TwoMomsOrder)
            for i in range(TwoMomsOrder):
                self.SteadyVector[i] = 1 / TwoMomsOrder
            #Parameters --> Gerhardt and Nelson 2009, pag 640 cv^2 > 1
            self.h2bAlpha = 0.5 * (1 + np.sqrt(1-2/(self.scv+1)))
            self.h2bRate = 2 * self.h2bAlpha / MeanSvTime
            
            #build T, zeta
            TMat[0,0] = -self.h2bRate
            TMat[1,1] = -self.h2bRate * (1 - self.h2bAlpha)/ self.h2bAlpha
            ZetaVec[0] = self.h2bAlpha
            ZetaVec[1] = 1 - self.h2bAlpha
            #invT for h2b is easy b/c T is diagonal
            for i in range(TwoMomsOrder):
                TMatInv[i,i] = 1/TMat[i,i]
        
        if self.scv > 1 or self.scv < 1:
            #calculate implied third moment
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    for l in range(TwoMomsOrder):
                        TMatSqInv[i,j] = TMatSqInv[i,j] + TMatInv[i,l] * TMatInv[l,j]
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    for l in range(TwoMomsOrder):
                        TMatCuInv[i,j] = TMatCuInv[i,j] + TMatInv[i,l] * TMatSqInv[l,j]
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    ImpThirdMoms = ImpThirdMoms - 6 * ZetaVec[i] * TMatCuInv[i,j]
        #Find minimum Markov-MECO order
        skew = (ImpThirdMoms - 3 * MeanSvTime * TrueSecMoms + 2 * np.power(MeanSvTime,3)/(np.power(TrueVarMoms,(3/2))))
        #Specify Markov-MECO to match
        if Coeffvar > 0 and skew >= Coeffvar - 1/Coeffvar:
            MinN1 = 1 / self.scv
            MinN2 = ((-skew + 1 / np.power(Coeffvar,3)+ 1/Coeffvar + 2*Coeffvar)/
                    (skew - (Coeffvar - 1/Coeffvar)))
            #Check if given order feasible
            if MinN1 > MinN2:
                k = int(MinN1) + 1
            else:
                k = int(MinN2) + 1
        else:
            print("Error Bad MECO")
            exit()
        NeedBiggerOrder = True
        while (NeedBiggerOrder):
            mecoM1 = MeanSvTime
            mecoM2 = TrueSecMoms
            mecoM3 = ImpThirdMoms
            mecoX = mecoM1* mecoM3 - ( (k+2) / (k+1) )*np.power(mecoM2,2)
            mecoY = mecoM2 - ( (k+1)/ k )* np.power(mecoM1,2)
            mecoA = k * (k+2) * mecoM1 * mecoY
            mecoB = -( k * mecoX + k * ( (k+2) / (k+1)) * np.power(mecoY,2) 
                      + (k+2) * mecoY * np.power(mecoM1,2) )
            mecoC = mecoM1 * mecoX
            mecoRoot1 = (-mecoB + np.sqrt( np.power(mecoB,2) - 4* mecoA * mecoC ) ) / (2*mecoA)
            mecoRoot2 = (-mecoB - np.sqrt( np.power(mecoB,2) - 4* mecoA * mecoC ) ) / (2*mecoA)
            
            MECOOrder = k
            self.MECORate1 = 1 / mecoRoot1
            MECORate2 = 1 / mecoRoot2
            self.MECOAlpha = ( (mecoM1 / MECOOrder) - mecoRoot2 ) / ( mecoRoot1 - mecoRoot2 )
            #Now get Markov - MECO stuff
            tempMMECOCov = TrueVarMoms * self.rho
            self.MMECOAlpha21 = self.MECOAlpha - tempMMECOCov / ( (1 - self.MECOAlpha)*
                                                       np.power((MECOOrder* mecoRoot1 - 
                                                       MECOOrder*mecoRoot2),2) )
            self.MMECOAlpha12 = self.MMECOAlpha21 * (1 - self.MECOAlpha) / self.MECOAlpha
            self.MMECORate1 = self.MECORate1
            self.MMECORate2 = MECORate2
            # Check feasibility of M-MECO probs
            if self.MMECOAlpha21 < 0 or self.MMECOAlpha21 > 1 or self.MMECOAlpha12 < 0 or self.MMECOAlpha12 > 1:
                k = k + 1
                if k > 4000:
                    print("Cannot match target moments, check feasible rho for this scv.")
                    exit()
            else:
                NeedBiggerOrder = False
                self.MMECOOrder = MECOOrder
        #Steady State Vector
        self.SteadyVector = np.empty(2*self.MMECOOrder)
        self.SteadyCumVector = np.empty(2*self.MMECOOrder)
        for i in range(self.MMECOOrder):
            self.SteadyVector[i] = (MECORate2 * self.MMECOAlpha21) / (self.MMECOOrder * (self.MECORate1 * self.MMECOAlpha12 + MECORate2 * self.MMECOAlpha21))
        for i in range(self.MMECOOrder, 2*self.MMECOOrder):
            self.SteadyVector[i] = (self.MECORate1 * self.MMECOAlpha12) / (self.MMECOOrder * (self.MECORate1 * self.MMECOAlpha12 + MECORate2 * self.MMECOAlpha21))
        #from survey paper
        MMECOCInf = self.scv * (1 + 2 * self.MECOAlpha * self.rho / self.MMECOAlpha21)

        # Model Message
        if self.show_m == True:
            print("Majorizing Process is Markov-MECO(%d)" %(self.MMECOOrder))
            print("Rates: %.2f, %.2f " %(self.MMECORate1, self.MMECORate2) )
            print("Prob21: %.4f, Prob12: %.4f" %(self.MMECOAlpha21,self.MMECOAlpha12) )
            print("IDC: %.2f" %(MMECOCInf) )

        #Cumulative steady-state prob vector
        self.SteadyCumVector[0] = self.SteadyVector[0]
        for i in range(1,2*self.MMECOOrder):
            self.SteadyCumVector[i] = self.SteadyVector[i] + self.SteadyCumVector[i - 1]
    
    def _start_rep(self, stream):
        """
        Starts every replication
        Parameters:
            Stream: A Stream method or function that generate the random varietes
        Return:
            Void
        """
        #Reinitialize the storage per replication
        self.interarr_storage = []
        self.arr_storage = []

        if (self.WithDSPP):
            if self.GammaN > 25:
                self.XVar = (1 / self.GammaN) * stream._GammaSpecOne(self.GammaN)
            else:
                self.XVar = (1 / self.GammaN) * stream._ErlangOneStep(self.GammaN, 1)
        else:
            self.XVar = 1
        self.LastS = self.XVar * self._my_cum_rate_func(self.sim_end_time)
        #Schedule first potential arrival
        if self.MajPhType == 1:           #exponential
            FirstArrTime = stream._expon(ExponMean)
        elif self.MajPhType == 2:       #h2b
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand() 
            if WhichBasePh < self.SteadyCumVector[0]:
                #initially in phase 1--generate exponential from lambda1
                FirstArrTime = stream._expon(1 / self.h2bRate)
            else:
                #initially in phase 2--generate exponential from lambda2
                FirstArrTime = stream._expon(1 / (self.h2bRate * (1 - self.h2bAlpha) / self.h2bAlpha))
        elif self.MajPhType == 3:          #MECon
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand()
            R = 0
            while True:
                if WhichBasePh < self.SteadyCumVector[R]:
                    break
                R = R + 1
            CurPhase = R
            if CurPhase < self.MEConOrder + 1:
                #initially in chain 1--generate Erlang of K+1-R
                FirstArrTime = stream._ErlangOneStep(self.MEConOrder + 1 - CurPhase, 1 / self.MEConRate)
            else:
                #initially in chain 2--generate Erlang of 2*K-R
                FirstArrTime = stream._ErlangOneStep(2 * self.MEConOrder - CurPhase, 1 / self.MEConRate)
        elif self.MajPhType == 4:   #MECO (3-moment)
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand()
            R = 0
            while True:
                if WhichBasePh < self.SteadyCumVector[R]:
                    break
                R = R + 1
            CurPhase = R
            if CurPhase < MECOOrder + 1:
                #initially in chain 1--generate Erlang of K+1-R, with rate1
                FirstArrTime = stream._ErlangOneStep(MECOOrder + 1 - CurPhase, 1 / self.MECORate1)
            else:
                #initially in chain 2--generate Erlang of 2*K+1-R, with rate2
                FirstArrTime = stream._ErlangOneStep(2 * MECOOrder + 1 - CurPhase, 1 / MECORate2)
        elif self.MajPhType == 5:      #Markov-MECO
            #WhichBasePh = lcgrand(CompMixProb)
            try:
                WhichBasePh = stream.rand()
            except AttributeError:
                WhichBasePh = stream.randomly.rand()
            R = 0
            while True:
                if WhichBasePh < self.SteadyCumVector[R]:
                    break
                R = R + 1
            CurPhase = R
            if CurPhase < self.MMECOOrder:
                #initially in chain 1--generate Erlang of K+1-R, with rate1
                FirstArrTime = stream._ErlangOneStep(self.MMECOOrder - CurPhase, 1 / self.MMECORate1)
                self.PrevGenErlang = 1
            else:
                #initially in chain 2--generate Erlang of 2*K+1-R, with rate2
                FirstArrTime = stream._ErlangOneStep(2 * self.MMECOOrder - CurPhase, 1 / self.MMECORate2)
                self.PrevGenErlang = 2
        if FirstArrTime > self.LastS:
            InvFirstTime = self.sim_end_time + 1
        else:
            InvFirstTime = self._my_inv_rate_func(FirstArrTime/self.XVar)
            self.interarr_storage.append(InvFirstTime)
            self.arr_storage.append(InvFirstTime)
        self.curr_time = InvFirstTime
    
    def _make_arrival(self,stream):
        """
        Generates the arrival given an arrival function
        Parameters:
            Stream: A Stream method or function that generate the random varietes
        Return:
            Void
        """ 
        # Schedule next potential arrival
        if self.MajPhType == 1:               #exponential
            NextArrTime = stream._expon(self.ExponMean)
        elif self.MajPhType == 2:           # h2b
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand()
            if WhichBasePh < self.h2bAlpha:
                NextArrTime = stream._expon(1 / self.h2bRate)
            else:
                NextArrTime = stream._expon(1 / (self.h2bRate * (1 - self.h2bAlpha) / self.h2bAlpha))
        elif self.MajPhType == 3 :           # MECon
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand()
            if WhichBasePh < 1 - self.MEConAlpha:
                NextArrTime = stream._ErlangOneStep(self.MEConOrder, 1 / self.MEConRate)
            else:
                NextArrTime = stream._ErlangOneStep(self.MEConOrder - 1, 1 / self.MEConRate)
        
        elif self.MajPhType == 4:           # MECO (3-moment)
            #WhichBasePh = lcgrand(CompMixProb)
            WhichBasePh = stream.expo_rand.rand()
            if WhichBasePh < self.MECOAlpha:
                NextArrTime = stream._ErlangOneStep(MECOOrder, 1 / self.MECORate1)
            else:
                NextArrTime = stream._ErlangOneStep(MECOOrder, 1 / MECORate2)
        
        elif self.MajPhType == 5:           # Markov-MECO
            #WhichBasePh = lcgrand(CompMixProb)
            try:
                WhichBasePh = stream.rand()
            except AttributeError:
                WhichBasePh = stream.randomly.rand()
            if self.PrevGenErlang == 1:
                if WhichBasePh < 1 - self.MMECOAlpha12:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate1)
                    self.PrevGenErlang = 1
                else:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate2)
                    self.PrevGenErlang = 2
            else:
                if WhichBasePh < self.MMECOAlpha21:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate1)
                    self.PrevGenErlang = 1
                else:
                    NextArrTime = stream._ErlangOneStep(self.MMECOOrder, 1 / self.MMECORate2)
                    self.PrevGenErlang = 2
        # Calculate NS arrival time: R^-1(NextArrTime)
        self.curr_time = self.XVar * self._my_cum_rate_func(self.curr_time) #sim_time por self.curr_time
        if self.curr_time + NextArrTime > self.LastS:
            InvTime = self.sim_end_time + 1
        else:
            InvTime = self._my_inv_rate_func((self.curr_time + NextArrTime) / self.XVar)
            self.interarr_storage.append(InvTime-self.arr_storage[-1])
            self.arr_storage.append(InvTime)
        self.curr_time = InvTime

    def get_rho_range(self):
        """Based on scv, get feasible range of rho"""
        MaxOrderSize = 50000
        self.CanMakeMMECO = True
        MeanSvTime = 1
        TrueVarMoms = self.scv * np.power(MeanSvTime,2)
        TrueSecMoms = TrueVarMoms + np.power(MeanSvTime,2)
        Coeffvar = np.sqrt(self.scv)
        ImpThirdMoms = 0 #Initialize Implied thid moment as zero
        # Get Implied Third Moment
        if self.scv == 1:
            # exponential
            TwoMomsOrder = 1
            ImpThirdMoms = 6
        elif self.scv < 1:
            # MECon
            # determine K
            k = 1
            while (1 / k > self.scv):
                k = k + 1
            # Parameters
            self.MEConOrder = k
            self.MEConAlpha = (1 / (1 + self.scv)) * (k * self.scv - np.sqrt(k * (1 + self.scv) - np.power(k,2) 
                                                                        * self.scv))
            self.MEConRate = (k - self.MEConAlpha) / MeanSvTime
            TwoMomsOrder = 2 * self.MEConOrder - 1
            TMat = np.zeros((TwoMomsOrder,TwoMomsOrder))  #this is D0 \equiv U(A1-I)
            TMatInv = np.zeros( (TwoMomsOrder,TwoMomsOrder) )
            TMatSqInv = np.zeros( (TwoMomsOrder,TwoMomsOrder))
            TMatCuInv = np.zeros( (TwoMomsOrder,TwoMomsOrder))
            ZetaVec= np.zeros(TwoMomsOrder)    # this is zeta vec
            ZetaVec[0] = 1 - self.MEConAlpha
            ZetaVec[self.MEConOrder] = self.MEConAlpha
            # TMatInv directly: relative easy
            for i in range(self.MEConOrder):
                for j in range(i,self.MEConOrder):
                    TMatInv[i, j] = - 1 / self.MEConRate
            for i in range(self.MEConOrder, 2 * self.MEConOrder - 1):
                for j in range(i, 2 * self.MEConOrder - 1):
                    TMatInv[i, j] = - 1 / self.MEConRate
        else:
            # h2b
            TwoMomsOrder = 2

            TMat = np.zeros((TwoMomsOrder,TwoMomsOrder))        # this is D0 \equiv U(A1-I)
            TMatInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            TMatSqInv = np.zeros((TwoMomsOrder,TwoMomsOrder)) 
            TMatCuInv = np.zeros((TwoMomsOrder,TwoMomsOrder))
            ZetaVec = np.zeros(TwoMomsOrder)     # this is zeta vec
            
            #get h2b params
            # Parameters
            self.h2bAlpha = 0.5 * (1 + np.sqrt(1 - 2 / (self.scv + 1)))
            self.h2bRate = 2 * self.h2bAlpha / MeanSvTime
            # build T, zeta
            TMat[0,0] = -self.h2bRate
            TMat[1,1] = -self.h2bRate * (1 - self.h2bAlpha) / self.h2bAlpha
            ZetaVec[0] = self.h2bAlpha
            ZetaVec[1] = 1 - self.h2bAlpha
            # invT for h2b is easy b/c T is diagonal
            for i in range(TwoMomsOrder):
                TMatInv[i, i] = 1 / TMat[i, i]

        if self.scv > 1 or self.scv < 1:
            # calc implied third moment
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    for k in range(TwoMomsOrder):
                        TMatSqInv[i, j] = TMatSqInv[i, j] + TMatInv[i, k] * TMatInv[k, j]
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    for k in range(TwoMomsOrder):
                        TMatCuInv[i, j] = TMatCuInv[i, j] + TMatInv[i, k] * TMatSqInv[k, j]
            for i in range(TwoMomsOrder):
                for j in range(TwoMomsOrder):
                    ImpThirdMoms = ImpThirdMoms - 6 * ZetaVec[i] * TMatCuInv[i, j]
        
        # find minimum Markov-MECO order
        skew = (ImpThirdMoms - 3 * MeanSvTime * TrueSecMoms + 2 * np.power(MeanSvTime,3) ) / np.power(TrueVarMoms,(3 / 2))

        #Specfy Markov-MECO to match
        if Coeffvar > 0 and skew >= Coeffvar - 1 / Coeffvar:
            MinN1 = 1 / self.scv
            MinN2 = (-skew + 1 / np.power(Coeffvar, 3) + 1 / Coeffvar + 2 * Coeffvar) / (skew - (Coeffvar - 1 / Coeffvar))

            # Check if given order feasible
            if MinN1 > MinN2:
                k = int(MinN1) + 1
            else:
                k = int(MinN2) + 1
        else:
            print("Bad MECO")
            self.CanMakeMMECO = False
            exit()

        if MinN1 > MinN2:
            InitK = int(MinN1) + 1
        else:
            InitK = int(MinN2) + 1

        CurMaxGuess = 1
        FoundMax = False

        # start looking for max rho1 achievable
        k = InitK
        while (FoundMax == False):
            NeedBiggerOrder = True
            while (NeedBiggerOrder):
                mecoM1 = MeanSvTime
                mecoM2 = TrueSecMoms
                mecoM3 = ImpThirdMoms

                mecoX = mecoM1 * mecoM3 - ((k + 2) / (k + 1)) * np.power(mecoM2,2)
                mecoY = mecoM2 - ((k + 1) / k) * np.power(mecoM1,2)
                mecoA = k * (k + 2) * mecoM1 * mecoY
                mecoB = -(k * mecoX + k * ((k + 2) / (k + 1)) * np.power(mecoY,2) + (k + 2) * mecoY * np.power(mecoM1,2))
                mecoC = mecoM1 * mecoX

                mecoRoot1 = (-mecoB + np.sqrt(np.power(mecoB,2) - 4 * mecoA * mecoC)) / (2 * mecoA)
                mecoRoot2 = (-mecoB - np.sqrt(np.power(mecoB,2) - 4 * mecoA * mecoC)) / (2 * mecoA)

                TempMECOOrder = k
                self.MECORate1 = 1 / mecoRoot1
                MECORate2 = 1 / mecoRoot2
                MECOAlpha = ((mecoM1 / TempMECOOrder) - mecoRoot2) / (mecoRoot1 - mecoRoot2)

                # now get Markov-MECO stuff
                tempMMECOCov = TrueVarMoms * CurMaxGuess
                self.MMECOAlpha21 = MECOAlpha - tempMMECOCov / ((1 - MECOAlpha) * np.power(
                                (TempMECOOrder * mecoRoot1 - TempMECOOrder * mecoRoot2),2) )
                self.MMECOAlpha12 = self.MMECOAlpha21 * (1 - MECOAlpha) / MECOAlpha
                self.MMECORate1 = self.MECORate1
                self.MMECORate2 = MECORate2
                # check feasibility of M-MECO probs
                if self.MMECOAlpha21 < 0 or self.MMECOAlpha21 > 1 or self.MMECOAlpha12 < 0 or self.MMECOAlpha12 > 1:
                    k = k + 1
                    if k > MaxOrderSize:
                        NeedBiggerOrder = False
                        CurMaxGuess = CurMaxGuess - 0.001
                        k = InitK
                else:
                    NeedBiggerOrder = False
                    FoundMax = True
        MaxRho1 = CurMaxGuess

        # start looking for min rho1 achievable
        CurMinGuess = MaxRho1
        FoundMin = False
        k = InitK
        while (FoundMin == False):
            NeedBiggerOrder = True
            while (NeedBiggerOrder):
                mecoM1 = MeanSvTime
                mecoM2 = TrueSecMoms
                mecoM3 = ImpThirdMoms
                mecoX = mecoM1 * mecoM3 - ((k + 2) / (k + 1)) * np.power(mecoM2,2)
                mecoY = mecoM2 - ((k + 1) / k) * np.power(mecoM1,2)
                mecoA = k * (k + 2) * mecoM1 * mecoY
                mecoB = -(k * mecoX + k * ((k + 2) / (k + 1)) * np.power(mecoY,2) + (k + 2) * mecoY * np.power(mecoM1,2) )
                mecoC = mecoM1 * mecoX

                mecoRoot1 = (-mecoB + np.sqrt(np.power(mecoB,2) - 4 * mecoA * mecoC)) / (2 * mecoA)
                mecoRoot2 = (-mecoB - np.sqrt(np.power(mecoB,2) - 4 * mecoA * mecoC)) / (2 * mecoA)

                TempMECOOrder = k
                self.MECORate1 = 1 / mecoRoot1
                MECORate2 = 1 / mecoRoot2
                MECOAlpha = ((mecoM1 / TempMECOOrder) - mecoRoot2) / (mecoRoot1 - mecoRoot2)

                # now get Markov-MECO stuff
                tempMMECOCov = TrueVarMoms * CurMinGuess
                self.MMECOAlpha21 = MECOAlpha - tempMMECOCov / ((1 - MECOAlpha) * np.power((TempMECOOrder * mecoRoot1 - TempMECOOrder * mecoRoot2),2) )
                self.MMECOAlpha12 = self.MMECOAlpha21 * (1 - MECOAlpha) / MECOAlpha
                self.MMECORate1 = self.MECORate1
                self.MMECORate2 = MECORate2
                # check feasibility of M-MECO probs
                if self.MMECOAlpha21 < 0 or self.MMECOAlpha21 > 1 or self.MMECOAlpha12 < 0 or self.MMECOAlpha12 > 1:
                    k = k + 1
                    if k > MaxOrderSize:
                        FoundMin = True
                        NeedBiggerOrder = False
                else:
                    NeedBiggerOrder = False
                    CurMinGuess = CurMinGuess - 0.001
                    k = InitK
        MinRho1 = CurMinGuess
        # Model String
        print("Feasible rho in [%.4f,%.4f]" %(MinRho1, MaxRho1))
        
        
        
class Streams(object):
    """
    Description: This class takes care of the creation of all distributions
    inside the simulation. 
    """
    def __init__(self, compmixprob, expo_seed, expo_serv):
        """
        This is the constructor of the class.
        
        Parameters:
            service_rand: A seed to implement CRN on the service time
            inter_seed: A seed to implement CRN on the interarrivals
        """
        self.randomly = RandomState()
        self.randomly.seed(compmixprob)
        self.expo_rand = RandomState()
        self.expo_rand.seed(expo_seed)
        self.expo_serv = RandomState()
        self.expo_serv.seed(expo_serv)

    def _GammaSpecOne(self,n):
        # Generate gamma(N,1)
        Sum = 0
        for i in range(n):
            Sum = Sum - np.log(self.expo_rand.rand())
        return Sum

    def _expon(self,Mean):
        #Function to generate exponential variates with mean Mean via inverse cdf
        return -np.log(1 - self.expo_rand.rand()) * Mean

    def _erlang(self,m, Mean):  
        # Erlang variate generation function
        Sum = 0
        for i in range(m):
            Sum = Sum + self._expon(Mean/m)
        return Sum

    def _ErlangOneStep(self,n, Beta):
        #Erlang generation in one step
        Prod = 1
        for i in range(n):
            Prod = Prod * (1 - self.expo_rand.rand())
        return - Beta * np.log(Prod) 
    
class RNG(object):
    def __init__(self, compmixprob, exponstream, expo_serv):
        self.MODLUS = 2147483647
        self.MULT1 = 24112
        self.MULT2 = 26143

        self.zrng = np.empty(100,dtype=int)
        self.zrng[0] = 1973272912
        self.zrng[1] = 281629770
        self.zrng[2] = 20006270
        self.zrng[3] = 1280689831
        self.zrng[4] = 2096730329
        self.zrng[5] = 1933576050
        self.zrng[6] = 913566091
        self.zrng[7] = 246780520
        self.zrng[8] = 1363774876
        self.zrng[9] = 604901985
        self.zrng[10] = 1511192140
        self.zrng[11] = 1259851944
        self.zrng[12] = 824064364
        self.zrng[13] = 150493284
        self.zrng[14] = 242708531
        self.zrng[15] = 75253171
        self.zrng[16] = 1964472944
        self.zrng[17] = 1202299975
        self.zrng[18] = 233217322
        self.zrng[19] = 1911216000
        self.zrng[20] = 726370533
        self.zrng[21] = 403498145
        self.zrng[22] = 993232223
        self.zrng[23] = 1103205531
        self.zrng[24] = 762430696
        self.zrng[25] = 1922803170
        self.zrng[26] = 1385516923
        self.zrng[27] = 76271663
        self.zrng[28] = 413682397
        self.zrng[29] = 726466604
        self.zrng[30] = 336157058
        self.zrng[31] = 1432650381
        self.zrng[32] = 1120463904
        self.zrng[33] = 595778810
        self.zrng[34] = 877722890
        self.zrng[35] = 1046574445
        self.zrng[36] = 68911991
        self.zrng[37] = 2088367019
        self.zrng[38] = 748545416
        self.zrng[39] = 622401386
        self.zrng[40] = 2122378830
        self.zrng[41] = 640690903
        self.zrng[42] = 1774806513
        self.zrng[43] = 2132545692
        self.zrng[44] = 2079249579
        self.zrng[45] = 78130110
        self.zrng[46] = 852776735
        self.zrng[47] = 1187867272
        self.zrng[48] = 1351423507
        self.zrng[49] = 1645973084
        self.zrng[50] = 1997049139
        self.zrng[51] = 922510944
        self.zrng[52] = 2045512870
        self.zrng[53] = 898585771
        self.zrng[54] = 243649545
        self.zrng[55] = 1004818771
        self.zrng[56] = 773686062
        self.zrng[57] = 403188473
        self.zrng[58] = 372279877
        self.zrng[59] = 1901633463
        self.zrng[60] = 498067494
        self.zrng[61] = 2087759558
        self.zrng[62] = 493157915
        self.zrng[63] = 597104727
        self.zrng[64] = 1530940798
        self.zrng[65] = 1814496276
        self.zrng[66] = 536444882
        self.zrng[67] = 1663153658
        self.zrng[68] = 855503735
        self.zrng[69] = 67784357
        self.zrng[70] = 1432404475
        self.zrng[71] = 619691088
        self.zrng[72] = 119025595
        self.zrng[73] = 880802310
        self.zrng[74] = 176192644
        self.zrng[75] = 1116780070
        self.zrng[76] = 277854671
        self.zrng[77] = 1366580350
        self.zrng[78] = 1142483975
        self.zrng[79] = 2026948561
        self.zrng[80] = 1053920743
        self.zrng[81] = 786262391
        self.zrng[82] = 1792203830
        self.zrng[83] = 1494667770
        self.zrng[84] = 1923011392
        self.zrng[85] = 1433700034
        self.zrng[86] = 1244184613
        self.zrng[87] = 1147297105
        self.zrng[88] = 539712780
        self.zrng[89] = 1545929719
        self.zrng[90] = 190641742
        self.zrng[91] = 1645390429
        self.zrng[92] = 264907697
        self.zrng[93] = 620389253
        self.zrng[94] = 1502074852
        self.zrng[95] = 927711160
        self.zrng[96] = 364849192
        self.zrng[97] = 2049576050
        self.zrng[98] = 638580085
        self.zrng[99] = 547070247

        self.rand_seed = compmixprob
        self.expo_seed = exponstream
        self.expo_serv = expo_serv

    def rand(self):
        #Stream = self.seed
        if self.rand_seed>99 or self.rand_seed<0:
            print("Stream out of range of Stream")
            exit()
        zi = self.zrng[self.rand_seed]
        lowprd = (zi & 65535) * self.MULT1
        hi31 = (zi >> 16) * self.MULT1 + (lowprd >> 16)
        zi = (((lowprd & 65535) - self.MODLUS) + 
            ((hi31 & 32767) << 16) + (hi31 >> 15))    
        if zi < 0:
            zi += self.MODLUS  
        lowprd = (zi & 65535) * self.MULT2
        hi31 = (zi >> 16) * self.MULT2 + (lowprd >> 16)
        zi = (((lowprd & 65535) - self.MODLUS) + 
            ((hi31 & 32767) << 16) + (hi31 >> 15))
        if zi < 0:
            zi += self.MODLUS
        self.zrng[self.rand_seed] = zi
        return (zi >> 7 | 1) / 16777216

    def lcgrand(self,stream):
        if stream>99 or stream<0:
            print("Stream out of range of Stream")
            exit()
        zi = self.zrng[stream]
        lowprd = (zi & 65535) * self.MULT1
        hi31 = (zi >> 16) * self.MULT1 + (lowprd >> 16)
        zi = (((lowprd & 65535) - self.MODLUS) + 
            ((hi31 & 32767) << 16) + (hi31 >> 15))    
        if zi < 0:
            zi += self.MODLUS  
        lowprd = (zi & 65535) * self.MULT2
        hi31 = (zi >> 16) * self.MULT2 + (lowprd >> 16)
        zi = (((lowprd & 65535) - self.MODLUS) + 
            ((hi31 & 32767) << 16) + (hi31 >> 15))
        if zi < 0:
            zi += self.MODLUS
        self.zrng[stream] = zi
        return (zi >> 7 | 1) / 16777216

    def lcgrandst(self,zset,stream):
        """ Set the current zrng for stream "stream" to zset. """
        self.zrng[stream] = zset

    def lcgrandgt(self,stream):
        """Return the current zrng for stream "stream"."""
        return self.zrng[stream]

    def Change_seed(self,new_rand_seed=-1,new_expo_seed=-1):
        """Helps to change the seed when running"""
        if new_rand_seed != -1:
            self.rand_seed = new_rand_seed
        if new_expo_seed != -1:
            self.expo_seed = new_expo_seed

    def _expon(self, Mean):
        # Function to generate exponential variates with mean Mean via inverse cdf
        return -np.log(1 - self.lcgrand(self.expo_seed)) * Mean

    def _serv(self, Mean):
        # Function to generate exponential variates with mean Mean via inverse cdf
        return -np.log(1 - self.lcgrand(self.expo_serv)) * Mean

    def _uniform(self, Lower, Upper,stream):
        """ 
        Function to generate U(Lower, Upper) variates via inverse cdf
        Parameters:
        Lower As Double, 
        Upper As Double, 
        seed As Integer
        """
        return Lower + (Upper - Lower) * self.lcgrand(stream)

    def _random_integer(self,prob_distrib,stream):
        """
        Function that generate a U(0,1) random variate an then return a random integer in accordance with the (accumulative) distribution
        Parameters: prob_distrib: distribution probability. (should be piecewise)
        seed: Seed to use
        return: observation of prob_distrib
        """
        U = self.lcgrand(stream)
        #  function prob_distrib.
        random_integer = 1
        while U >= prob_distrib(random_integer):
            random_integer = random_integer + 1

    def _erlang(self,m, Mean): 
        """ 
        Erlang variate generation function. 
        Parameters:
        m: As Integer, 
        Mean: As Double, 
        Return: An erlang random observation
        """
        mean_exponential = Mean / m
        Sum = 0
        for i in range(m):
            Sum = Sum + self._expon(mean_exponential)
        return Sum

    def _ErlangOneStep(self,n, Beta):
        """
        Erlang generation in one step
        Parameters:
        n As Integer,
        Beta As Double, 
        """
        Prod = 1
        for i in range(n):
            U = self.lcgrand(self.expo_seed)
            Prod = Prod * (1 - U)
        return -Beta * np.log(Prod)

    def _GammaSpecOne(self,n):
        """ 
        Generate gamma(N,1)
        Parameters:
        n As Integer, 
        """
        Sum = 0
        for i in range(n):
            U = self.lcgrand(self.expo_seed)
            Sum = Sum - np.log(U)  
        return Sum