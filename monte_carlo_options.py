import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
S0    = 100.0    # Current stock price
K     = 105.0    # Strike price
r     = 0.05     # Risk-free rate (annual)
sigma = 0.20     # Volatility (annual)
T     = 1.0      # Time to expiry (years)
dt    = 1/252    # Daily steps
n_steps       = int(T / dt)
n_simulations = 100_000

# ── Simulate risk-neutral price paths (drift = r, not mu) ─────────────────────
np.random.seed(42)
Z = np.random.standard_normal((n_simulations, n_steps))
daily_returns = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

paths = np.ones((n_simulations, n_steps + 1)) * S0
for t in range(n_steps):
    paths[:, t + 1] = paths[:, t] * daily_returns[:, t]

final_prices = paths[:, -1]

# ── Monte Carlo option prices ─────────────────────────────────────────────────
discount = np.exp(-r * T)

call_payoffs = np.maximum(final_prices - K, 0)
put_payoffs  = np.maximum(K - final_prices, 0)

mc_call  = discount * np.mean(call_payoffs)
mc_put   = discount * np.mean(put_payoffs)
mc_call_se = discount * np.std(call_payoffs) / np.sqrt(n_simulations)
mc_put_se  = discount * np.std(put_payoffs)  / np.sqrt(n_simulations)

# ── Black-Scholes analytical prices ──────────────────────────────────────────
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

bs_call = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
bs_put  = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# ── Print results ─────────────────────────────────────────────────────────────
print("=== European Option Pricing via Monte Carlo ===")
print(f"S0={S0}  K={K}  r={r:.0%}  σ={sigma:.0%}  T={T:.0f}yr  N={n_simulations:,}")
print()
print(f"{'':20s} {'Call':>10s}  {'Put':>10s}")
print("-" * 44)
print(f"{'Monte Carlo':20s} {'${:.4f}'.format(mc_call):>10s}  {'${:.4f}'.format(mc_put):>10s}")
print(f"{'  95% CI ±':20s} {'${:.4f}'.format(1.96*mc_call_se):>10s}  {'${:.4f}'.format(1.96*mc_put_se):>10s}")
print(f"{'Black-Scholes':20s} {'${:.4f}'.format(bs_call):>10s}  {'${:.4f}'.format(bs_put):>10s}")
print(f"{'Error':20s} {'${:.4f}'.format(abs(mc_call-bs_call)):>10s}  {'${:.4f}'.format(abs(mc_put-bs_put)):>10s}")

# ── Convergence: price vs number of simulations ───────────────────────────────
sample_sizes = np.logspace(2, 5, 40).astype(int)
conv_call, conv_put = [], []
for n in sample_sizes:
    fp = paths[:n, -1]
    conv_call.append(discount * np.mean(np.maximum(fp - K, 0)))
    conv_put.append( discount * np.mean(np.maximum(K - fp, 0)))

# ── Plots ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Sample paths
ax = axes[0]
time_axis = np.linspace(0, T, n_steps + 1)
ax.plot(time_axis, paths[:300].T, alpha=0.08, linewidth=0.6, color="steelblue")
ax.axhline(K,  color="red",   linestyle="--", linewidth=1.5, label=f"Strike K=${K}")
ax.axhline(S0, color="black", linestyle=":",  linewidth=1.2, label=f"S0=${S0}")
ax.set_title("Simulated Price Paths (300)")
ax.set_xlabel("Time (years)")
ax.set_ylabel("Price ($)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2) Payoff distributions
ax = axes[1]
itm_calls = call_payoffs[call_payoffs > 0]
itm_puts  = put_payoffs[put_payoffs > 0]
ax.hist(itm_calls, bins=60, alpha=0.6, color="green",  label=f"Call payoffs (ITM {len(itm_calls)/n_simulations:.0%})")
ax.hist(itm_puts,  bins=60, alpha=0.6, color="tomato", label=f"Put payoffs  (ITM {len(itm_puts)/n_simulations:.0%})")
ax.axvline(mc_call / discount, color="green",  linestyle="--", linewidth=2, label=f"Mean call payoff ${mc_call/discount:.2f}")
ax.axvline(mc_put  / discount, color="tomato", linestyle="--", linewidth=2, label=f"Mean put payoff  ${mc_put/discount:.2f}")
ax.set_title("In-the-Money Payoff Distributions")
ax.set_xlabel("Payoff at Expiry ($)")
ax.set_ylabel("Frequency")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 3) Convergence
ax = axes[2]
ax.semilogx(sample_sizes, conv_call, color="green",  linewidth=2, label="MC Call")
ax.semilogx(sample_sizes, conv_put,  color="tomato", linewidth=2, label="MC Put")
ax.axhline(bs_call, color="green",  linestyle="--", linewidth=1.5, label=f"BS Call ${bs_call:.4f}")
ax.axhline(bs_put,  color="tomato", linestyle="--", linewidth=1.5, label=f"BS Put  ${bs_put:.4f}")
ax.set_title("MC Price Convergence to Black-Scholes")
ax.set_xlabel("Number of Simulations")
ax.set_ylabel("Option Price ($)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(f"Monte Carlo European Options — S0=${S0}, K=${K}, r={r:.0%}, σ={sigma:.0%}, T={T:.0f}yr", fontsize=12)
plt.tight_layout()
plt.savefig("monte_carlo_options.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved to monte_carlo_options.png")
