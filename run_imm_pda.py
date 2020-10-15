# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from gaussparams import GaussParams
from mixturedata import MixtureParameters
import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
import estimationstatistics as estats


from utils import track_and_evaluate_sequence, TrackResult
import utils

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
# if "inline" in matplotlib.get_backend():
#     print("Plotting is set to inline at the moment:", end=" ")

#     if "ipykernel" in matplotlib.get_backend():
#         print("backend is ipykernel (IPython?)")
#         print("Trying to set backend to separate window:", end=" ")
#         import IPython

#         IPython.get_ipython().run_line_magic("matplotlib", "")
#     else:
#         print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "data_for_imm_pda.mat"
loaded_data = scipy.io.loadmat(filename_to_load)
K = loaded_data["K"].item()
Ts = loaded_data["Ts"].item()
Xgt = loaded_data["Xgt"].T
Z = [zk.T for zk in loaded_data["Z"].ravel()]
true_association = loaded_data["a"].ravel()
time = np.arange(K) * Ts

# plot measurements close to the trajectory
fig1, ax1 = plt.subplots(num=1, clear=True)

Z_plot_data = np.empty((0, 2), dtype=float)
plot_measurement_distance = 45
for Zk, xgtk in zip(Z, Xgt):
    to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
    Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

ax1.scatter(*Z_plot_data.T, s=5, color="C1")
ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
ax1.set_title("True trajectory and the nearby measurements")
fig1.tight_layout()
fig1.savefig("figs/sim_trajectory.pdf")
plt.show(block=False)

# %% play measurement movie. Remember that you can cross out the window
play_movie = False
play_slice = slice(0, K)
if play_movie:
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    fig2, ax2 = plt.subplots(num=2, clear=True)
    sh = ax2.scatter(np.nan, np.nan, s=5)
    th = ax2.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax2.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = 0.1
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig2.canvas.draw_idle()
        plt.show(block=False)
        plt.pause(plotpause)


# %% IMM-PDA

# THE PRESET PARAMETERS AND INITIAL VALUES WILL CAUSE TRACK LOSS!
# Some reasoning and previous exercises should let you avoid track loss.
# No exceptions should be generated if PDA works correctly with IMM,
# but no exceptions do not guarantee correct implementation.

# sensor
sigma_z = 1.8
clutter_intensity = 0.002
PD = 0.999
gate_size = 3

# dynamic models
sigma_a_CV = 0.01
sigma_a_CT = 0.3
sigma_omega = 0.005*np.pi


# markov chain
PI11 = 0.95
PI22 = 0.95

p10 = 0.5  # initvalue for mode probabilities

PI = np.array([[PI11, (1 - PI11)], [(1 - PI22), PI22]])
assert np.allclose(np.sum(PI, axis=1), 1), "rows of PI must sum to 1"

# not valid

mean_init = np.array([0, 0, 0, 0, 0])
cov_init = np.diag([50, 50, 10, 10, 0.1]) ** 2  
mode_probabilities_init = np.array([p10, (1 - p10)])
mode_states_init = GaussParams(mean_init, cov_init)
init_imm_state = MixtureParameters(mode_probabilities_init, [mode_states_init] * 2)

assert np.allclose(
    np.sum(mode_probabilities_init), 1
), "initial mode probabilities must sum to 1"

# make model
measurement_model = measurementmodels.CartesianPosition(sigma_z, state_dim=5)
dynamic_models: List[dynamicmodels.DynamicModel] = []
dynamic_models.append(dynamicmodels.WhitenoiseAccelleration(sigma_a_CV, n=5))
dynamic_models.append(dynamicmodels.ConstantTurnrate(sigma_a_CT, sigma_omega))
ekf_filters = []
ekf_filters.append(ekf.EKF(dynamic_models[0], measurement_model))
ekf_filters.append(ekf.EKF(dynamic_models[1], measurement_model))
imm_filter = imm.IMM(ekf_filters, PI)

modes = ["CV", "CT"]

tracker = pda.PDA(imm_filter, clutter_intensity, PD, gate_size)
trackresult = track_and_evaluate_sequence(tracker, init_imm_state, Z, Xgt, Ts, K)

# consistency
confprob = 0.9
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))
CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K

utils.write_csv_results(trackresult, confprob, "figs/sim")

# %% plots
# trajectory
fig3, axs3 = plt.subplots(1, 2, num=3, clear=True)
utils.trajectory_plot(axs3[0], trackresult, Xgt)
utils.mode_scatter(axs3[0], trackresult, 1)
utils.mode_plot(axs3[1], trackresult, time, labels=modes)
fig3.tight_layout()
fig3.savefig("figs/sim_modeplot.pdf")

# NEES
NEESpos = trackresult.NEESpos
fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
utils.confidence_interval_plot(axs4[0], time, trackresult.NEESpos, CI2, confprob, "NEES pos")
utils.confidence_interval_plot(axs4[1], time, trackresult.NEESvel, CI2, confprob, "NEES vel")
utils.confidence_interval_plot(axs4[2], time, trackresult.NEES, CI4, confprob, "NEES")
fig4.tight_layout()
fig4.savefig("figs/sim_neess.pdf")

ANEESpos = trackresult.ANEESpos
ANEESvel = trackresult.ANEESvel
ANEES = trackresult.ANEES
print(f"ANEESpos = {ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEESvel = {ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
print(f"ANEES = {ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

# errors
fig5, axs5 = plt.subplots(2, num=5, clear=True)
axs5[0].plot(np.arange(K) * Ts, trackresult.pos_error)
axs5[0].set_ylabel("position error")
axs5[1].plot(np.arange(K) * Ts, trackresult.vel_error)
axs5[1].set_ylabel("velocity error")
fig5.tight_layout()
fig5.savefig("figs/sim_errorplot.pdf")

plt.show()
