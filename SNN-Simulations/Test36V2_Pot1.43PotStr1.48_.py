
# coding: utf-8

# In[ ]:

import nest
#import nest.raster_plot
#import pylab
import time
from numpy import exp
import numpy
import math
import random

Inizio=time.time()

class ImportIniLIFCA():
    #inizializzo le informazioni da cercare in perseo.ini
    inf=["NeuronType",               #ancora valore fisso
         "DelayDistribType",         #ancora valore fisso
         "SynapticExtractionType",   #ancora valore fisso
         "Life"] 
    
    def __init__(self,files):
        self.files=files
    
    def FilesControllo(self):
        import sys
        for i in range(0,len(self.files)):
            if self.FileControllo(self.files[i]):
                sys.exit(0)

    def FileControllo(self,file1):
        try:
            f1=open(file1,"r")
            f1.close()
            return 0
        except ValueError:
            print "ValueError"
            return 1
        except IOError as err:
            print("OS error: {0}".format(err))   
            return 1
        except:
            print("Unexpected error:", sys.exc_info()[0])
            return 1    
        
    def Estrai_inf(self,stampa=0):

        InfoPerseo=self.EstraiInfoPerseo()                   #estrai le info da perseo.ini
        AppoggioTempM=self.EstraiInfoModuli()                #estrai le info da modules.ini
        AppoggioTempC=self.EstraiInfoConnectivity()          #estrai le info da connectivity.ini
        AppoggioTempP=self.EstraiProtocol()                  #estrai le info da protocol.ini

        def getKey(item):
            return item[0]
        #InfoProtocol=sorted(AppoggioTempP,key=getKey)
        InfoProtocol=AppoggioTempP        
        # converto le informazioni estratte in un formato adatto da tuple a lista
        
        InfoBuildT=[AppoggioTempM[0]]
        for i in range(0,AppoggioTempM[0]):
            app1=[int(AppoggioTempM[2][i][0])]
            app=(app1+list(AppoggioTempM[2][i][3:9])+list(AppoggioTempM[2][i][12])+list(AppoggioTempM[2][i][9:12]))
            InfoBuildT.append(app)
        del app    

        InfoBuild=[float(InfoBuildT[0])]
        for i in range(0,int(InfoBuildT[0])):
            app=[]
            for j in range(0,11):
                app.append(float(InfoBuildT[i+1][j]))
            InfoBuild=InfoBuild+[app]
        del app

        InfoConnectPop=[AppoggioTempM[0]]
        for i in range(0,len(AppoggioTempC[1][:])):
            app=list(AppoggioTempC[1][i])
            InfoConnectPop.append(app)
        del app

        InfoConnectNoise=[AppoggioTempM[0]]
        for i in range(0,AppoggioTempM[0]):
            app=list(AppoggioTempM[2][i][1:3])
            InfoConnectNoise.append(app)
            
            
        if stampa==1: #stampa a schermo dei dati salvati
            for i,j in enumerate(InfoPerseo):
                print self.inf[i],"=",j
            print
            print "La rete è composta da %d popolazioni di neuroni" % AppoggioTempM[0]
            print AppoggioTempM[1]
            for i in range(0,AppoggioTempM[0]):
                print AppoggioTempM[2][i]
            print
            print AppoggioTempC[0]
            for i in range(0,AppoggioTempM[0]**2):
                print AppoggioTempC[1][i]
            print
            for i in InfoProtocol:
                print "SET_PARAM"+str(i)


        return InfoPerseo,InfoBuild,InfoConnectPop,InfoConnectNoise,InfoProtocol
    
    def EstraiProtocol(self):
        import string
        f1=open(self.files[3],"r")  
        ProtocolList= []
        for x in f1.readlines():    
            y=string.split(x) 
            if len(y):
                if x[0]!="#" and y[0]=="SET_PARAM":
                    try:
                        ProtocolList.append([float(y[1]),int(y[2]),float(y[3]),float(y[4])])                      
                    except ValueError:                        
                        pass                                       
        f1.close()
        return ProtocolList
                                            
    def EstraiInfoPerseo(self):
        import string
        f1=open(self.files[0],"r")  
        InfList= []
        for x in f1.readlines():    
            y=string.split(x) 
            if len(y):
                if x[0]!="#":
                    for findinf in self.inf:
                        try:
                            temp=y.index(findinf)
                            InfList.append(y[temp+2])                      
                        except ValueError:                        
                            pass                                       
        f1.close()
        return InfList

    def EstraiInfoModuli(self):
        import string
        f1=open(self.files[2],"r")  
        NumPop=0
        for i,x in enumerate(f1.readlines()):    
            y=string.split(x) 
            if len(y):
                if x[0]!="#":
                    NumPop=NumPop+1  
            if i==2:
                ParamList=[]
                for j in range(1,14):
                    ParamList.append(y[j])              
        f1.close()
        PopsParamList=[]
        f1=open(self.files[2],"r")
        x=f1.readlines()
        for j in range(0,NumPop):       
            #print string.split(x[4+j])
            PopsParamList.append(string.split(x[4+j]))
        f1.close()
        return NumPop,ParamList,PopsParamList
    
    def EstraiInfoConnectivity(self):
        import string
        f1=open(self.files[1],"r")      
        PopConParamList=[]
        for i,x in enumerate(f1.readlines()):    
            y=string.split(x) 
            if len(y):
                if x[0]!="#":
                    PopConParamList.append(y)
            if i==1:
                ParamList=[]
                for j in range(1,9):
                    ParamList.append(y[j]) 
        f1.close()
        return ParamList,PopConParamList

#----------------------------------------------------------------------------------------------------------------------    
    

Salva=1
#num=21
#PotLevRange=numpy.linspace(1.35, 1.5, num)
#PotLevRange=numpy.concatenate((numpy.linspace(1.35, 1.5, num)[::-1],numpy.linspace(1.3575, 1.5, num-1)))
#-->

for ContApp in range(0,1,1):  


    #File.ini con i dati da importare per la simulaizone, 
    #l'ordine in cui sono scritti e la loro formattazione è rilevante
    file1="perseo35.ini"
    file2="c_cortsurf_Pot1.43PotStr148v3.ini"
    file3="m_cortsurf_Pot1.43.ini"
    #file2="/media/sf_condivisa/Test Milano/connectivity.EI.ini"
    #file3="/media/sf_condivisa/Test Milano/modules.EI.ini"
    file4="ProtocolExploration36_Kick26.ini"
    files=[file1,file2,file3,file4]
    
    FileName="dati/Rates_Nest_Run_Milano_Test36_13x13_"+str(nest.Rank())+"_Pot1.43PotStr148v3Long3.dat"
    #controllo l'esistenza dei file in lettura       
    ImpFil=ImportIniLIFCA(files);
    ImpFil.FilesControllo()

    #estraggo le informazione di interesse dai files.ini e le trasferisco sui files:
    #InfoPerseo,InfoBuild,InfoConnectPop,InfoConnectNoise

    stampa=0;  #stampa=1 produce in output i dati della simulazione stampa=0 no
    InfoPerseo,InfoBuild,InfoConnectPop,InfoConnectNoise,InfoProtocol=ImpFil.Estrai_inf(stampa) 

    # InfoPerseo=["NeuronType","DelayDistribType","SynapticExtractionType","Life" ]
    # InfoBuild=[numero di popolazioni,
    #             [N,C_ext,\nu_ext,\tau,\tetha,H,\tau_arp,NeuronInitType,\alpha_c,\tau_c,g_c],
    #             [.....],[],...]
    # InfoConnectPop=[numero di popolazioni,
    #             [post,pre,c,Dmin,Dmax,syn typ,J,DJ],
    #             [.....],[],...]
    # InfoConnectNoise=[numero di popolazioni,
    #             [J_ext,DJ_ext],
    #             [.....],[],...]
    # InfoProtocol=[[time,population,param_num,value],
    #             [.....],[],...]




    for iterazioni in range(0,1,1):
        #print "sto iterando :",iterazioni
        #print "al livello di potenziamento :",ContatoreEsterno

       
        if ContApp==0 and iterazioni==0:
            #############################------------------------------------------------------------------------
            #Pulisco la rete
            #############################------------------------------------------------------------------------
            nest.ResetKernel()

            #############################------------------------------------------------------------------------
            #inserisco i parametri introduttivi della simulazione
            #############################------------------------------------------------------------------------


            dt = 0.1                                 # the resolution in ms
            StartMisure=1.                            # elimino dalla misurazione iprimi 1 ms
            simtime = int(float(InfoPerseo[3]))       # Simulation time in ms
                                                      # simtime=3000
            if simtime<=StartMisure:                  # Se il tempo di simulazione è inferiore al secondo viene aumentato di un secondo
                simtime=simtime+StartMisure           # Aumento condizionato del tempo di simulazione
            start=0.0                                 # Setto il riferimento temporale
            origin=0.0                                # Setto l'origine temporale

            #############################------------------------------------------------------------------------
            #Definisco il vettore Equilibri
            #############################------------------------------------------------------------------------
            ''' 
            Equilibri=[] 
            for i in range(0,int(InfoBuild[0])):
                Equilibri.append([])
                for j in range(0,2*num-1):
                    Equilibri[i].append([])
            ''' 
            #############################------------------------------------------------------------------------
            #Parametri Kernel
            #############################------------------------------------------------------------------------
            #nest.SetKernelStatus({"total_num_virtual_procs": 90})
            nest.SetKernelStatus({"local_num_threads": 31})#25
            nest.SetKernelStatus({"resolution": dt, "print_time": False,
                                  "overwrite_files": True})

            #############################------------------------------------------------------------------------
            #"randomizzo" i seed dei generatori random 
            #############################------------------------------------------------------------------------

            # msd = int(math.fabs(time.clock()*1000))
			
            # N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
            # GrNg = nest.GetKernelStatus(['grng_seed'])[0]
            # print N_vp,GrNg
            # print nest.GetKernelStatus()			
            # pyrngs = [numpy.random.RandomState(s) for s in range(msd, msd+N_vp)]
            # nest.SetKernelStatus({"grng_seed" : msd+N_vp})
            # nest.SetKernelStatus({"rng_seeds" : range(msd+N_vp+1, msd+2*N_vp+1)})
            # nest.SetKernelStatus({"off_grid_spiking" : True})			
			
            # N_vp = nest.GetKernelStatus(['total_num_virtual_procs'])[0]
            # GrNg = nest.GetKernelStatus(['grng_seed'])[0]
            # print N_vp,GrNg
            # print nest.GetKernelStatus()				


            #############################------------------------------------------------------------------------
            print("Building network")
            #############################------------------------------------------------------------------------

            startbuild = time.time() #inizializzo il calcolo del tempo utilizzato per simulare

            NeuronPop=[]
            NoisePop=[]
            DetectorPop=[]
            
            #Definisco ed Inizializzo le popolazioni di neuroni con i parametri estratti dai file.ini
            for i in range(1,int(InfoBuild[0])+1):    
                if int(InfoBuild[i][7])==0:
                    app=float(InfoBuild[i][5])
                else:
                    app=0.   
                app2= nest.Create("aeif_psc_exp", int(InfoBuild[i][0]),params={"C_m":     1.0,
                                                                               "g_L":     1.0/float(InfoBuild[i][3]),
                                                                               "t_ref":   float(InfoBuild[i][6]),
                                                                               "E_L":     0.0,
                                                                               "V_reset": float(InfoBuild[i][5]),
                                                                               "V_m":     app,
                                                                               "V_th":    float(InfoBuild[i][4]),
                                                                               "Delta_T": 0.,
                                                                               "tau_syn_ex": 1.0,
                                                                               "tau_syn_in": 1.0,
                                                                               "a":     0.0,
                                                                               "b":     float(InfoBuild[i][10]),
                                                                               "tau_w": float(InfoBuild[i][9]),
                                                                               "V_peak":float(InfoBuild[i][4])+10.0})
                NeuronPop.append(app2)
                
            #Definisco ed Inizializzo i di generatori di poisson e gli spike detector con 
            #i parametri estratti dai file.ini
            for i in range(1,int(InfoBuild[0])+1):       
                app3= nest.Create("poisson_generator",params={"rate": float(InfoBuild[i][1]*InfoBuild[i][2]),
                                                             'origin':0.,
                                                             'start':start})
                NoisePop.append(app3)
                app4 = nest.Create("spike_detector",params={ "withtime": True,
                                                             "withgid": True,
                                                             "to_file": False,
															 "to_memory":True,
                                                           # "flush_after_simulate":True,#############!!!!ATTENTO
                                                           # "close_after_simulate":True,#############!!!!ATTENTO
                                                             "start":StartMisure})
                DetectorPop.append(app4)

            #endbuild = time.time()
            endbuild = time.time()
            #############################------------------------------------------------------------------------
            print("Connecting devices")
            #############################------------------------------------------------------------------------

            startconnect = time.time()
            Connessioni=[]
            Medie=[]
            
            #Creo e Definisco le connessioni tra le popolazioni di neuroni ed i generatori di poisson e 
            #tra le popolazioni di neuroni e gli spike detector con i parametri estratti dai file.ini
            for i in range(0,int(InfoBuild[0])):
            #for i in range(0,3):    

                nest.Connect(NoisePop[i], NeuronPop[i], syn_spec={'model': 'static_synapse', 
                                                          'delay': dt,        
                                                          'weight': {'distribution': 'normal_clipped',
                                                          'mu': float(InfoConnectNoise[i+1][0]),
                                                          'sigma': (float(InfoConnectNoise[i+1][1])*float(InfoConnectNoise[i+1][0])), 
                                                          'low': 0.,'high': float('Inf')} })
                nest.Connect(NeuronPop[i][:int(InfoBuild[i+1][0])], DetectorPop[i], syn_spec={"weight": 1.0, "delay": dt})

            #Creo e Definisco le connessioni tra le popolazioni di neuroni
            #con i parametri estratti dai file.ini
            for i in range(0,len(InfoConnectPop[1:])):
            #for i in range(0,1):
                ''' 
                nest.Connect(NeuronPop[int(InfoConnectPop[i+1][1])], NeuronPop[int(InfoConnectPop[i+1][0])],
                             {'rule': 'pairwise_bernoulli','p':float(InfoConnectPop[i+1][2]) })
                ''' 
                nest.Connect(NeuronPop[int(InfoConnectPop[i+1][1])], NeuronPop[int(InfoConnectPop[i+1][0])],#{'rule': 'all_to_all'}, 
                                                 #{'rule': 'pairwise_bernoulli',
                                                  #'p'::float(InfoConnectPop[i+1][2]) },   
                                                 {"rule": "fixed_indegree", "indegree":int(round(float(InfoConnectPop[i+1][2])*float(int(InfoBuild[1+ int(InfoConnectPop[i+1][1])][0]))))},												  
                                                  syn_spec={'model': 'static_synapse_hpc',      
                                                  'delay': {'distribution': 'exponential_clipped', 
                                                   'low': float(InfoConnectPop[i+1][3])-dt/2,
                                                   'high': float(InfoConnectPop[i+1][4]),
                                                   'lambda':float(2.99573227355/(float(InfoConnectPop[i+1][4])-float(InfoConnectPop[i+1][3])))}, 
                                                    'weight': {'distribution': 'normal',
                                                   'mu': float(InfoConnectPop[i+1][6]),
                                                   'sigma': math.fabs(0*float(InfoConnectPop[i+1][6])*float(InfoConnectPop[i+1][7]))} })

                
                #Connessioni.append(nest.GetConnections(NeuronPop[int(InfoConnectPop[i+1][1])],NeuronPop[int(InfoConnectPop[i+1][0])]))
                #Medie.append(float(InfoConnectPop[i+1][6]))
            endconnect = time.time()
        
        #############################------------------------------------------------------------------------
        print("Simulating")
        #############################------------------------------------------------------------------------   
		###################################################################################################################################################################
        if Salva:
            print("I m going to save the data")
            x=str(iterazioni)
            #FileName="dati/spikes_Nest_Run_Milano_Test21.dat"
            f = open(FileName,"w") 
            if len(InfoProtocol):
				print("I m going to split the simulation")
				tempo=0
				for contatore in range(0,len(InfoProtocol)):
					appoggio1=int((tempo+InfoProtocol[contatore][0])/1000.)
					appoggio2=int(tempo/1000.)
					appoggio3=tempo+InfoProtocol[contatore][0]
					if (appoggio1-appoggio2)>=1:
						T1=(1+appoggio2)*1000-tempo
						nest.Simulate(T1) 
						#SALVA I DATI!!!!
						###########################################################		
						Equilibri=[]
						for i in range(0,int(InfoBuild[0])):
							#events = nest.GetStatus(DetectorPop[i], "n_events")[0]
							#rate = events / (simtime-StartMisure) * 1000.0 / int(InfoBuild[i+1][0])
							#print "Pop",i," mean firing rate   : %.3f Hz" % rate
							Equilibri.append([])
							a=nest.GetStatus(DetectorPop[i])[0]["events"]["times"]
							if len(a)>0:
								Trange=(1000*int(numpy.min(a)/1000.),1000*int(numpy.min(a)/1000.)+1000)
								hist,Tbin=numpy.histogram(a,200,(Trange[0],Trange[1]))
								Equilibri[i]=hist*1000./(5.*int(InfoBuild[i+1][0]))
							else:
								Trange=(1000*int(tempo/1000.),1000*int(tempo/1000.)+1000)
								hist=numpy.zeros(200)
								Tbin=numpy.linspace(Trange[0],Trange[1],num=201)
								Equilibri[i]=hist
							#print  "Pop Num="+str(i)+"   ",Trange[0],Trange[1],numpy.min(a),numpy.max(a),len(hist),len(Equilibri[i])
							#b=nest.GetStatus(DetectorPop[i])[0]["events"]["senders"]   
							#nest.GetStatus(Info.SpikeDetectors[i]['Ndx'],keys='n_events')[0]
							#for j in range(0,len(a)):
							#	f.write(str(b[j]-1)+" "+str(a[j])+"\n")
							nest.SetStatus(DetectorPop[i],{'n_events':0})
						for j in range(0,len(hist)):
							f.write(str(Tbin[j])+" ")
							for i in range(0,int(InfoBuild[0])):
								#print "Saving pop="+str()+" timebin="+str(j)+"  "+str(Equilibri[i][j])
								f.write(str(Equilibri[i][j])+" ")
							f.write("\n ")	
						###########################################################
						tempo=tempo+T1
						for contatore2 in range(1,(appoggio1-appoggio2)):
							nest.Simulate(1000.) 
							#SALVA I DATI!!!!
							###########################################################		
							Equilibri=[]
							for i in range(0,int(InfoBuild[0])):
								#events = nest.GetStatus(DetectorPop[i], "n_events")[0]
								#rate = events / (simtime-StartMisure) * 1000.0 / int(InfoBuild[i+1][0])
								#print "Pop",i," mean firing rate   : %.3f Hz" % rate
								Equilibri.append([])
								a=nest.GetStatus(DetectorPop[i])[0]["events"]["times"]
								if len(a)>0:
									Trange=(1000*int(numpy.min(a)/1000.),1000*int(numpy.min(a)/1000.)+1000)
									hist,Tbin=numpy.histogram(a,200,(Trange[0],Trange[1]))
									Equilibri[i]=hist*1000./(5.*int(InfoBuild[i+1][0]))
								else:
									Trange=(1000*int(tempo/1000.),1000*int(tempo/1000.)+1000)
									hist=numpy.zeros(200)
									Tbin=numpy.linspace(Trange[0],Trange[1],num=201)
									Equilibri[i]=hist
								nest.SetStatus(DetectorPop[i],{'n_events':0})
							for j in range(0,len(hist)):
								f.write(str(Tbin[j])+" ")
								for i in range(0,int(InfoBuild[0])):
									f.write(str(Equilibri[i][j])+" ")
								f.write("\n ")	
							tempo=tempo+1000.
						T2=appoggio3-tempo
						nest.Simulate(T2);
						tempo=tempo+T2;
					else:
						nest.Simulate(InfoProtocol[contatore][0])
						temp=InfoProtocol[contatore][0]
						tempo=tempo+temp
					if InfoProtocol[contatore][2]==4:
						nest.SetStatus(NoisePop[InfoProtocol[contatore][1]],params={"rate": float(InfoBuild[1+InfoProtocol[contatore][1]][2]*InfoProtocol[contatore][3])})
					if InfoProtocol[contatore][2]==12:
						nest.SetStatus(NeuronPop[InfoProtocol[contatore][1]], params={"b": float(InfoProtocol[contatore][3])})
            else:
				nest.Simulate(simtime)
				tempo=simtime
            if (simtime-tempo)>0.:
				nest.Simulate(simtime-tempo)
				
				
            endsimulate = time.time()
            f.close()
        else:
			if len(InfoProtocol):
				tempo=0
				for contatore in range(0,len(InfoProtocol)):
					nest.Simulate(InfoProtocol[contatore][0]) 
					temp=InfoProtocol[contatore][0]
					tempo=tempo+temp
					if InfoProtocol[contatore][2]==4:
									nest.SetStatus(NoisePop[InfoProtocol[contatore][1]],params={"rate": float(InfoBuild[1+InfoProtocol[contatore][1]][2]*InfoProtocol[contatore][3])})
	#								print "Population:", InfoProtocol[contatore][1] ,";Parameter:", InfoProtocol[contatore][2]  ,";  Value: ",InfoProtocol[contatore][3]
					if InfoProtocol[contatore][2]==12:
												  nest.SetStatus(NeuronPop[InfoProtocol[contatore][1]], params={"b": float(InfoProtocol[contatore][3])})
	#											  print "Population:", InfoProtocol[contatore][1] ,";Parameter:", InfoProtocol[contatore][2]  ,";  Value: ",InfoProtocol[contatore][3]
					
			else:
				nest.Simulate(simtime)
				tempo=simtime
			if (simtime-tempo)>0.:
				nest.Simulate(simtime-tempo)
			endsimulate = time.time()


		###################################################################################################################################################################

        #############################------------------------------------------------------------------------
        #stampo i risultati della simulazione
        #############################------------------------------------------------------------------------

        num_synapses = nest.GetDefaults('static_synapse_hpc')["num_connections"] 
        build_time = endbuild - startbuild
        connect_time = endconnect - startconnect
        sim_time = endsimulate - endconnect

        N_neurons=0
        for i in range(0,int(InfoBuild[0])):
            N_neurons=N_neurons+int(InfoBuild[i+1][0])

        print" Network simulation (Python) neuron type:",InfoPerseo[0]
        print("Number of neurons : {0}".format(N_neurons))
        print("Number of synapses: {0}".format(num_synapses))
        print("Building time     : %.2f s" % build_time)
        print("Connecting time     : %.2f s" % connect_time)
        print("Simulation time   : %.2f s" % sim_time)

        #############################------------------------------------------------------------------------
        #scrivo su file  gli spikes
        #############################------------------------------------------------------------------------    
		
###################################################################################################################################################################
        # if Salva:
            # x=str(iterazioni)
            # FileName="dati/spikes_Nest_Run_Milano_Test9V2_Extended.dat"
            # f = open(FileName,"w") 

            # for i in range(0,int(InfoBuild[0])):
                # events = nest.GetStatus(DetectorPop[i], "n_events")[0]
                # rate = events / (simtime-StartMisure) * 1000.0 / int(InfoBuild[i+1][0])
                # print "Pop",i," mean firing rate   : %.3f Hz" % rate
                # #############################------------------------------------------------------------------------              
                # #Equilibri[i][ContApp].append(rate)    
                # #############################------------------------------------------------------------------------ 
                # a=nest.GetStatus(DetectorPop[i])[0]["events"]["times"]
                # b=nest.GetStatus(DetectorPop[i])[0]["events"]["senders"]     
                # for j in range(0,len(a)):
                    # f.write(str(b[j]-1)+" "+str(a[j])+"\n")
                # #if rate>0.0:
                # #    nest.raster_plot.from_device(DetectorPop[i], hist=True,title="Raster plot ", hist_binwidth=100.)

            # f.close() 
        # else:
            # for i in range(0,int(InfoBuild[0])):
                # events = nest.GetStatus(DetectorPop[i], "n_events")[0]
                # rate = events / (simtime-StartMisure) * 1000.0 / int(InfoBuild[i+1][0])
                # print "Pop",i," mean firing rate   : %.3f Hz" % rate
                # #############################------------------------------------------------------------------------              
                # #Equilibri[i][ContApp].append(rate)    
                # #############################------------------------------------------------------------------------ 
                # #a=nest.GetStatus(DetectorPop[i])[0]["events"]["times"]
                # #b=nest.GetStatus(DetectorPop[i])[0]["events"]["senders"]
###################################################################################################################################################################            

          
#import matplotlib.pyplot as plt
#plt.figure(1)
#plt.hold(True)
#for i in range(0,int(InfoBuild[0])):
#    plt.plot(PotLevRange,numpy.mean(Equilibri[i], axis=1), 'g^--')
#
#plt.plot([1.35,1.5],[3,3], 'k--',[1.35,1.5],[6,6], 'k--')
#    
'''    
plt.plot(PotLevRange,numpy.mean(Equilibri[0], axis=1), 'g^--',
         PotLevRange,numpy.mean(Equilibri[1], axis=1),'ro--',
         PotLevRange,numpy.mean(Equilibri[2], axis=1), 'r^--',
         PotLevRange,numpy.mean(Equilibri[3], axis=1), 'rs--',
         PotLevRange,numpy.mean(Equilibri[4], axis=1), 'bs--',
         [1.25,1.35],[3,3], 'k--',[1.25,1.35],[6,6], 'k--')
'''    

#plt.show()   
Fine=time.time()              
print ("Total Simulation time   : %.2f s" % (Fine-Inizio))

