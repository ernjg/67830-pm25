import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmdstanpy as sp
import numpy as np
import os

master_data = pd.read_json("./data/2023/janfebmaster.json")
normalize = 200
K = 8
fcst = master_data["pm25_geo1"].values / normalize
fcst = np.append(fcst, master_data["pm25_geo3"].values[-2:] / normalize)

stan_data = {
    "N": len(master_data),
    "K": K,
    "y": master_data["pm25_obs"].values.astype(float),
    "fcst": fcst.astype(float),
}

model = sp.CmdStanModel(stan_file=os.path.join("./data/linreg.stan"))
fit = model.sample(data=stan_data, show_progress=True, show_console=False)

d = master_data
pm25_obs = d["pm25_obs"].values
mu_draws = fit.stan_variable("mu_pred")
mu_mean = mu_draws.mean(axis=0)

t = np.arange(len(pm25_obs))
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(t, pm25_obs, label="pm25_obs", linewidth=1, color="C0")
ax.plot(t[K - 1 :], mu_mean[K - 1 :], label="model (posterior mean μ)", linewidth=1.2, color="C1")
ax.set_xlabel("Time index")
ax.set_ylabel("PM2.5 (µg/m³)")
ax.set_title("Observed PM2.5 vs Stan linear lag model")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
out_path = os.path.join("linreg_fit_plot.png")
plt.savefig(out_path, dpi=150)
print(f"Saved plot to {out_path}")
