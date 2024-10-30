using Plots,Dierckx,ProgressMeter,Statistics,LaTeXStrings,LinearAlgebra,StatsBase,Polyester,Interpolations,PoissonRandom,DelimitedFiles
include("NeuralFields.jl")


function getMUA(ν,c₁,c₂)
    return ν + c₁.*randn(length(ν)) .+ c₂
end


function SubSampledRate(ν,dt,Δt,Npop)
    νss=zeros(Npop)
    @batch for n=1:Npop
        νss[n]=sum(ν[:,n].*dt)/Δt
    end
    return νss
end

function smooth_matrix(M::Matrix{Float64}, k::Int)
    # Perform Singular Value Decomposition
    U, S, V = svd(M)
    # Set all but the first `k` singular values to zero
    S_smoothed = zeros(size(S))
    for i in 1:k
        S_smoothed[i] = S[i]
    end
    # Reconstruct the smoothed matrix
    M_smoothed = U * Diagonal(S_smoothed) * V'
    
    return M_smoothed
end


function CentralIndex(Npop,γ)
    Ind=1:3:Npop 
    Exc=zeros(length(Ind))
    Np=Int(round(sqrt(length(Ind))))
    M=reshape(Exc,(Np,Np))
    M[1+γ:Np-γ,1+γ:Np-γ].=1
    Boundary=findall(x-> x>0,M)
   
    Indx=LinearIndices(M)[Boundary]
    Exc=zeros(length(Ind))

    return 3 .*Indx .-2

end


function Frame(M,i)
    cmap=cgrad([colorant"rgb(78,56,55)",colorant"rgb(167,119,119)",colorant"rgb(223,218,174)"])

    rows, cols = size(M)

    # Define the original grid indices (1-based for Julia matrices)
    x = Float64.(1:rows)
    y = Float64.(1:cols)

    # Create an interpolation object with cubic splines on a scaled grid
    itp = Spline2D(x, y, M)
    fine_x = LinRange(1.0, Float64(rows), 100)  # Finer grid for interpolation on x-axis
    fine_y = LinRange(1.0, Float64(cols), 100)  # Finer grid for interpolation on y-axis

    # Generate the interpolated surface data
    Z = [itp(xi, yi) for xi in fine_x, yi in fine_y]
    X=range(0,stop=5,length=100)
    Y=range(0,stop=5,length=100)
    
    heatmap(X, Y, Z, legend=false,clims=(-0.2,2),title="time=  "*string(round(i*Δt,digits=2))*" [s]",cmap=cmap,xlabel="X [mm]",ylabel="Y [mm]",size=(300,300))


end


## 

NV=200 # Number of grid ponints 
α=0.5  # Vmin=-α*θ
dt=0.0005 #[s]
Δt=0.001 #[s]
δμ=0.02
Scale=1.0
cm,S,μx,σ2x,g,Npop,KJ,KJ2=NeuralFields.InitializeFPfromPerseusNet("m_cortsurf.ini","c_cortsurf.ini",NV,dt,α,Scale);

FS=true
Life=60 #[s]
ResStep=Int(round(Δt/dt)) # Resolution time step  [s]
steps=Int(round(Life/dt))
tt=range(0.0, stop=Life,length=steps)

ν,νt,c=zeros(ResStep,Npop),zeros(Npop),zeros(Npop)
Ix=zeros(Npop)

StimulationStartingTime=range(1,Life-1,step=1)
IndxStimulated=rand(CentralIndex(Npop,3),length(StimulationStartingTime)) # Index of population in the center of the field

StimulationStartingIndx = [argmin(abs.(tt .- value)) for value in StimulationStartingTime]
StimulationStoppingIndx =StimulationStartingIndx .+Int(round(0.005/dt))

ϵ=0.125

# Set position in plane (g,νₓ)
g[1:3:end].=g[1:3:end].*(1+ϵ)
μx[1:3:end].=μx[1:3:end].*(1-ϵ)
σ2x[1:3:end].=σ2x[1:3:end].*(1-ϵ)
μxF,σ2xF=copy(μx),copy(σ2x) #Reference value


kk=1
ss=0
RateFile=open("./rates.dat","w")
ProtocolFile=open("./Protocol.dat","w")
γ=10 #Strengt of perturbation
@showprogress for n=2:steps
    kk+=1
    # Compute the recurrent contribution
    μ= KJ*νt + μx -g.*c  +Ix # Mean of synaptic currents
    σ2=KJ2*νt +σ2x       # Variance of synaptic currents
    # c is the adaptation variable dc/dt =-c(t)/τC +ν(t)
    
    # Check if is time to start stimulating
    if n ∈ StimulationStartingIndx
        ss+=1
        μx[IndxStimulated[ss]]=μxF[IndxStimulated[ss]].*γ
        σ2x[IndxStimulated[ss]]=σ2xF[IndxStimulated[ss]].*γ
    end
    # Check if is time to stop stimulating
    if n ∈ StimulationStoppingIndx
        μx[IndxStimulated[ss]]=μxF[IndxStimulated[ss]]
        σ2x[IndxStimulated[ss]]=σ2xF[IndxStimulated[ss]]
    end
    
   @batch for i=1:Npop
        NeuralFields.IntegrateFP!(cm[i],μ[i], σ2[i],S[i],FS)
        ν[kk,i],νt[i],c[i]=S[i][cm[i].NV+1:cm[i].NV+3]
   end
   #save sub-sampled firing rate
   if kk == ResStep
        νss=SubSampledRate(ν,dt,Δt,Npop)
        writedlm(RateFile,vcat([n*dt],νss)')
        kk=0
   end
   
end
close(RateFile)
close(ProtocolFile)


## Open raster
d=readdlm("./rates.dat")
t,ν=d[:,1],d[:,2:end]

# Create animation of field dynamics
Δt=t[4]-t[3]
tstart=j+ 0.9
IndStart=Int(round(tstart/Δt))
IndEnd=IndStart +Int(round(12/Δt))

@gif for i=IndStart:30:IndEnd
    M=smooth_matrix(reshape(log10.(getMUA(ν[i,1:3:end],0.01,0.1)),(13,13)),13)
    Frame(M,i)    
end

