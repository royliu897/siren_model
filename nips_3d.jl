using DrWatson
@quickactivate

using Distributed, Sunny, WGLMakie, HDF5, LinearAlgebra, Random
addprocs(4)

@everywhere using DrWatson, Sunny, LinearAlgebra, HDF5, Random

# -------------------
# Crystal & Hamiltonian
# -------------------
@everywhere function nips_crystal()
    latvecs = lattice_vectors(5.8196, 10.084, 6.8959, 90, 106.221, 90)
    positions = [[0, 1/3, 0]]
    Crystal(latvecs, positions, 12)
end

@everywhere function generate_parameters(; factor=2, offset=1/2)
    Ax = -0.01
    Az = 0.21
    J1a = -2.7 * (1 + factor*(rand()-offset))
    J1b = -2.0 * (1 + factor*(rand()-offset))
    J2a = 0.2
    J2b = 0.2
    J3a = 13.9 * (1 + factor*(rand()-offset))
    J3b = 13.9 * (1 + factor*(rand()-offset))
    J4 = -0.38
    return (; Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4)
end

@everywhere function nips_system(; Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4)
    crystal = nips_crystal()
    sys = System(crystal, [1 => Moment(; s=1, g=2)], :dipole_uncorrected)
    S = spin_matrices(Inf)
    set_onsite_coupling!(sys, S -> Ax*S[1]^2 + Az*S[3]^2, 1)
    set_exchange!(sys, J1a, Bond(2, 1, [0,0,0]))
    set_exchange!(sys, J1b, Bond(2, 3, [0,0,0]))
    set_exchange!(sys, J2a, Bond(2,2,[1,0,0]))
    set_exchange!(sys, J2b, Bond(1,4,[0,0,0]))
    set_exchange!(sys, J3a, Bond(1,3,[0,0,0]))
    set_exchange!(sys, J3b, Bond(2,3,[1,0,0]))
    set_exchange!(sys, J4, Bond(1,1,[0,0,1]))
    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=10_000)
    return sys
end

# -------------------
# Parameters
# -------------------
nsamples = 100        # number of Hamiltonians
npoints = 200         # random points per Hamiltonian

# Define grid for h,k,l,w (full for sampling)
h_range = range(-1, 1, 50)
k_range = range(-1, 1, 50)
l_range = range(-1, 1, 50)
w_range = range(0, 150, 100)

# -------------------
# Generate and save
# -------------------
all_data = zeros(Float32, nsamples * npoints, 9) # 9 = 4 hamiltonian + 3 hkl + 1 w + 1 S

pmap(1:nsamples) do i
    done = false
    while !done
        try
            # Generate Hamiltonian and system
            params = generate_parameters()
            sys = nips_system(; params...)

            # Spin wave measurement
            measure = ssf_perp(sys; formfactors=[1 => FormFactor("Ni2")])
            swt = SpinWaveTheory(sys; measure=measure, regularization=1e-6)

            # Sample random points from the 4D grid
            h_idx = rand(1:length(h_range), npoints)
            k_idx = rand(1:length(k_range), npoints)
            l_idx = rand(1:length(l_range), npoints)
            w_idx = rand(1:length(w_range), npoints)

            for j in 1:npoints
                q_lab = [h_range[h_idx[j]], k_range[k_idx[j]], l_range[l_idx[j]]]
                q = inv(sys.crystal.recipvecs) * q_lab
                energy = w_range[w_idx[j]]
                intensity = intensities(swt, [q]; energies=[energy], kernel=gaussian(fwhm=4))[1]
                # store as: Ax, Az, J1a, J1b, h, k, l, w, S
                all_data[(i-1)*npoints + j, :] = Float32[
                    params.Ax, params.Az, params.J1a, params.J1b,
                    q_lab[1], q_lab[2], q_lab[3], energy, intensity
                ]
            end
            done = true
        catch e
            println("Sample failed, retrying...")
            println(e)
        end
    end
end

# Save everything in one file
outdir = joinpath(ENV["WORK"], "data_generate", "4d_data")
mkpath(outdir)
filepath = joinpath(outdir, "nips_100h_200pts.h5")

h5open(filepath, "w") do file
    write(file, "data", all_data)
end

println("Saved all data to $filepath")
