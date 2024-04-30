using ModelingToolkit, Plots, StatsPlots, DifferentialEquations, Turing, LinearAlgebra

COLOR_SCHEME = [
    colorant"#00A98F", # biolizard green
    colorant"#FFC000", # gold
    colorant"#1565A9", # blue
    colorant"#C00000", # red
    colorant"#0D0D0D"  # black
]

function lotka_volterra(; name, α, β, δ, γ, N, P)
    @parameters α = α β = β δ = δ γ = γ
    @variables t N(t) = N P(t) = P
    d = Differential(t);
    eqs = [
        d(N) ~ α*N - β*N*P,
        d(P) ~ δ*N*P - γ*P
    ]
    return ODESystem(eqs, t; name)
end
 
a = lotka_volterra(name = :dyneq, α = 1.5, β = 0.01, δ = 0.01, γ = 0.8, N = 100, P = 15)
a = structural_simplify(a)
prob = ODEProblem(a, [], (0, 12), [])
sol = solve(prob, Tsit5(); saveat=0.1)
plot(
    sol,
    lw=2,
    label=["Klebsiella" "bacteriophage"],
    color=COLOR_SCHEME[1:2]',
    alpha=0.3,
    xlabel="Time (h)", ylabel="Population",
    legend=:topright,
    fmt=:svg, dpi=500, size=(600, 600)
)
#savefig("figures/LV_dyn_eq.svg")

# draw noisy oberservations from LV model
odedata = Array(sol) + 20 * randn(size(Array(sol)))

# Plot simulation and noisy observations.
scatter!(sol.t, odedata'; color=COLOR_SCHEME[1:2]', label="")

# model in Turing
@model function fitlv(data, prob)
    # Prior distributions.
    σ ~ InverseGamma(2, 3)
    α ~ truncated(Normal(1.5, 0.5), 0.5, 2.5)
    β ~ truncated(Normal(0.01, 0.01), 0, 2)
    γ ~ truncated(Normal(0.8, 0.3), 1, 4)
    δ ~ truncated(Normal(0.01, 0.01), 0, 2)

    # Simulate Lotka-Volterra model. 
    p = [α, β, γ, δ]
    predicted = solve(prob, Tsit5(); p=p, saveat=0.1)

    # Observations.
    for i in 1:length(predicted)
        data[:, i] ~ MvNormal(predicted[i], σ^2 * I)
    end

    return nothing
end

model = fitlv(odedata, prob)

# Sample 3 independent chains with forward-mode automatic differentiation (the default).
chain = sample(model, NUTS(0.65), MCMCSerial(), 10000, 3; progress=false)
plot(chain)

plot(; legend=false)
posterior_samples = sample(chain[[:α, :β, :γ, :δ]], 300; replace=false)
for p in eachrow(Array(posterior_samples))
    sol_p = solve(prob, Tsit5(); p=p, saveat=0.1)
    plot!(sol_p; alpha=0.1, color="#BBBBBB")
end
display(plot!())
# Plot simulation and noisy observations.
plot!(sol; color=COLOR_SCHEME[1:2]', linewidth=1)
scatter!(sol.t, odedata'; color=COLOR_SCHEME[1:2]')

# extermination example
b = lotka_volterra(name = :cool, α = 1.2, β = 0.01, δ = 0.02, γ = 0.08, N = 100, P = 15)
b = structural_simplify(b)
prob2 = ODEProblem(b, [], (0,12), [])
sol2 = solve(prob2)
plot(
    sol2,
    lw=2,
    label=["Klebsiella" "bacteriophage"],
    color=COLOR_SCHEME[1:2]',
    xlabel="Time (h)", ylabel="Population",
    legend=:topright,
    fmt=:svg, dpi=500, size=(600, 600)
)
savefig("figures/LV_extermination.svg")