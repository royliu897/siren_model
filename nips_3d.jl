using DrWatson
@quickactivate

using Distributed, Sunny, WGLMakie
addprocs(4)

@everywhere using DrWatson
@everywhere @quickactivate
@everywhere using Sunny, LinearAlgebra, HDF5, Random

# --- Crystal and system setup ---
@everywhere function nips_crystal()
    latvecs = lattice_vectors(5.8196, 10.084, 6.8959, 90, 106.221, 90)
    positions = [[0, 1/3, 0]]
    Crystal(latvecs, positions, 12)
end

# --- Generate LHS Hamiltonians ---
@everywhere function lhs_hamiltonians(n_ham::Int=50, factor::Float64=2.0)
    # ranges for the 4 varying parameters
    ranges = [
        (-2.7*(1-factor/2), -2.7*(1+factor/2)),  # J1a
        (-2.0*(1-factor/2), -2.0*(1+factor/2)),  # J1b
        (13.9*(1-factor/2), 13.9*(1+factor/2)),  # J3a
        (13.9*(1-factor/2), 13.9*(1+factor/2))   # J3b
    ]
    step = 1 / n_ham
    lhs_points = zeros(n_ham, 4)
    for i in 1:4
        lhs_points[:,i] = (randperm(n_ham) .- rand(n_ham)) .* step
        # scale to actual parameter range
        a,b = ranges[i]
        lhs_points[:,i] .= a .+ lhs_points[:,i] .* (b - a)
    end
    return lhs_points
end

# --- Build system ---
@everywhere function nips_system(; Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4)
    crystal = nips_crystal()
    sys = System(crystal, [1 => Moment(; s=1, g=2)], :dipole_uncorrected)
    S = spin_matrices(Inf)
    set_onsite_coupling!(sys, S -> Ax*S[1]^2 + Az*S[3]^2, 1)

    set_exchange!(sys, J1a, Bond(2, 1, [0, 0, 0]))
    set_exchange!(sys, J1b, Bond(2, 3, [0, 0, 0]))

    set_exchange!(sys, J2a,  Bond(2, 2, [1, 0, 0]))
    set_exchange!(sys, J2b,  Bond(1, 4, [0, 0, 0]))

    set_exchange!(sys, J3a,  Bond(1, 3, [0, 0, 0]))
    set_exchange!(sys, J3b,  Bond(2, 3, [1, 0, 0]))

    set_exchange!(sys, J4,  Bond(1, 1, [0, 0, 1]))

    randomize_spins!(sys)
    minimize_energy!(sys; maxiters=10_000)
    return sys
end

# --- Parameters ---
N_HAM = 50           # number of Hamiltonians
N_COORDS = 100_000   # points per Hamiltonian
Ax, Az, J2a, J2b, J4 = -0.01, 0.21, 0.2, 0.2, -0.38
outdir = joinpath(ENV["WORK"], "data_generate", "data")
mkpath(outdir)

# --- Parallel loop ---
lhs_points = lhs_hamiltonians(N_HAM)
pmap(1:N_HAM) do i
    done = false
    while !done
        try
            J1a, J1b, J3a, J3b = lhs_points[i,:]
            params = (; Ax, Az, J1a, J1b, J2a, J2b, J3a, J3b, J4)
            sys = nips_system(; params...)

            measure = ssf_perp(sys; formfactors=[1 => FormFactor("Ni2")])
            swt = SpinWaveTheory(sys; measure=measure, regularization=1e-6)

            # Sample coordinates randomly
            qa = rand(N_COORDS) .* 2 .- 1
            qb = rand(N_COORDS) .* 2 .- 1
            qc = rand(N_COORDS) .* 2 .- 1
            E  = rand(N_COORDS) .* 150

            qs_lab = [ [a,b,c] for (a,b,c) in zip(qa,qb,qc) ]

            # Compute intensities
            kernel = gaussian(; fwhm=4)
            rotations = [([0,0,1], π/3), ([0,0,1], 2π/3), ([0,0,1], 0.0)]
            weights = [1,1,1]

            res = domain_average(sys.crystal, qs_lab; rotations=rotations, weights=weights) do path_rotated
                intensities(swt, path_rotated; energies=E, kernel)
            end

            # Save to H5
            filename = "J1a=$(round(J1a,digits=3))_J1b=$(round(J1b,digits=3))_J3a=$(round(J3a,digits=3))_J3b=$(round(J3b,digits=3)).h5"
            filepath = joinpath(outdir, filename)
            h5open(filepath, "w") do f
                f["qa"] = qa
                f["qb"] = qb
                f["qc"] = qc
                f["E"]  = E
                f["J1a"] = fill(J1a, N_COORDS)
                f["J1b"] = fill(J1b, N_COORDS)
                f["J3a"] = fill(J3a, N_COORDS)
                f["J3b"] = fill(J3b, N_COORDS)
                f["data"] = res.data
            end

            done = true
        catch e
            println("\nFAILED SAMPLE. Retrying...\n")
            println(e)
        end
    end
end
