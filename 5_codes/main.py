from npm_core import compute_density, compute_ceff, compute_critical_density

# Example: 100B tokens, medium domain breadth, moderate reasoning depth
rho = compute_density(N_B=100, sigma=0.5, alpha=1.5)
rho_c = compute_critical_density(sigma=0.5, tau=2.0, k=10, delta=1.0)
c_eff = compute_ceff(rho, rho_c, beta=1.4)

print(f"Data density: {rho:.1f}")
print(f"Critical density: {rho_c:.1f}")
print(f"Generalization capacity: {c_eff:.1f}")