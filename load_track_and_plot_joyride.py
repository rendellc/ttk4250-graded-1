# %% imports
from typing import List

import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import dynamicmodels
import measurementmodels
import ekf
import imm
import pda
from gaussparams import GaussParams
from mixturedata import MixtureParameters
import estimationstatistics as estats

import utils
from utils import track_and_evaluate_sequence, TrackResult

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

# print("continuing with this plotting backend", end="\n\n\n")


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

def play_measurement_movie(fig, ax, play_slice, Z, dt):
    if "inline" in matplotlib.get_backend():
        print("the movie might not play with inline plots")
    sh = ax.scatter(np.nan, np.nan)
    th = ax.set_title(f"measurements at step 0")
    mins = np.vstack(Z).min(axis=0)
    maxes = np.vstack(Z).max(axis=0)
    ax.axis([mins[0], maxes[0], mins[1], maxes[1]])
    plotpause = dt
    plt.show(block=False)
    # sets a pause in between time steps if it goes to fast
    for k, Zk in enumerate(Z[play_slice]):
        sh.set_offsets(Zk)
        th.set_text(f"measurements at step {k}")
        fig.canvas.draw_idle()
        plt.pause(plotpause)


def evaluate_on_joyride(tracker, init_state, do_play_estimation_movie = False, start_k = 0, end_k = 10, plotfileprefix = ""):
    Z, Xgt, K, Ts = utils.load_pda_data("data_joyride.mat")
    assert len(Z) == K
    assert len(Z) == len(Xgt)

    # plot measurements close to the trajectory
    fig1, ax1 = plt.subplots(num=1, clear=True)
    Z_plot_data = np.empty((0, 2), dtype=float)
    plot_measurement_distance = 45
    for Zk, xgtk in zip(Z, Xgt):
        to_plot = np.linalg.norm(Zk - xgtk[None:2], axis=1) <= plot_measurement_distance
        Z_plot_data = np.append(Z_plot_data, Zk[to_plot], axis=0)

    ax1.scatter(*Z_plot_data.T, color="C1")
    ax1.plot(*Xgt.T[:2], color="C0", linewidth=1.5)
    ax1.set_title("True trajectory and the nearby measurements")
    fig1.tight_layout()
    fig1.savefig(plotfileprefix+"true_trajectory.eps")

    # %% play measurement movie. Remember that you can cross out the window
    do_play_measurement_movie = False
    play_slice = slice(0, K)
    if do_play_measurement_movie:
        fig2, ax2 = plt.subplots(num=2, clear=True)
        play_measurement_movie(fig2, ax2, play_slice, Z, 0.1)


    # First measurement is time 0 -> don't predict before first update.
    Ts = [0, *Ts]
    trackresult = track_and_evaluate_sequence(tracker, init_state, Z, Xgt, Ts, K)

    tr = trackresult # shorter name

    # consistency
    confprob = 0.9
    CI2 = np.array(scipy.stats.chi2.interval(confprob, 2))
    CI4 = np.array(scipy.stats.chi2.interval(confprob, 4))
    CI2K = np.array(scipy.stats.chi2.interval(confprob, 2 * K)) / K
    CI4K = np.array(scipy.stats.chi2.interval(confprob, 4 * K)) / K

    time = np.cumsum(Ts)

    if isinstance(tracker.state_filter, imm.IMM):
        print("Tracker uses IMM")

        # %% IMM-tracker plots
        # trajectory
        fig, axs3 = plt.subplots(1, 2, num=3, clear=True)
        utils.trajectory_plot(axs3[0], trackresult, Xgt)
        # probabilities
        utils.mode_plot(axs3[1], trackresult, time)

        # NEES
        fig, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
        utils.confidence_interval_plot(axs4[0], time, tr.NEESpos, CI2, confprob, "NEES pos")
        utils.confidence_interval_plot(axs4[1], time, tr.NEESvel, CI2, confprob, "NEES vel")
        utils.confidence_interval_plot(axs4[2], time, tr.NEES, CI4, confprob, "NEES")

        print(f"ANEESpos = {tr.ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEESvel = {tr.ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEES = {tr.ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

        # errors
        fig5, axs5 = plt.subplots(2, num=5, clear=True)
        axs5[0].plot(time, tr.pos_error)
        axs5[0].set_ylabel("position error")
        axs5[1].plot(time, tr.vel_error)
        axs5[1].set_ylabel("velocity error")

        if do_play_estimation_movie:
            play_estimation_movie(tracker, Z, trackresult.predict_list, trackresult.update_list, start_k, end_k)

    elif isinstance(tracker.state_filter, ekf.EKF):
        print("Tracker uses EKF")

        # %% EKF-tracker plots
        fig, ax = plt.subplots(num=3, clear=True)
        utils.trajectory_plot(ax, trackresult, Xgt)

        fig4, axs4 = plt.subplots(3, sharex=True, num=4, clear=True)
        utils.confidence_interval_plot(axs4[0], time, tr.NEESpos, CI2, confprob, "NEES pos")
        utils.confidence_interval_plot(axs4[1], time, tr.NEESvel, CI2, confprob, "NEES vel")
        utils.confidence_interval_plot(axs4[2], time, tr.NEES, CI4, confprob, "NEES")
        fig4.tight_layout()

        print(f"ANEESpos = {tr.ANEESpos:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEESvel = {tr.ANEESvel:.2f} with CI = [{CI2K[0]:.2f}, {CI2K[1]:.2f}]")
        print(f"ANEES = {tr.ANEES:.2f} with CI = [{CI4K[0]:.2f}, {CI4K[1]:.2f}]")

        fig5, axs5 = plt.subplots(2, num=5, clear=True)
        axs5[0].plot(time, tr.pos_error)
        axs5[0].set_ylabel("position error")
        axs5[1].plot(time, tr.vel_error)
        axs5[1].set_ylabel("velocity error")
    else:
        print("Invalid tracker type")


def play_estimation_movie(tracker, Z, predict_list, update_list, start_k, end_k):
    # %% TBD: estimation "movie"

    mTL = 0.2  # maximum transparancy (between 0 and 1);
    plot_pause = 1  # lenght to pause between time steps;
    plot_range = slice(start_k, end_k)  # the range to go through

    # %k = 31; assert(all([k > 1, k <= K]), 'K must be in proper range')
    fig6, axs6 = plt.subplots(1, 2, num=6, clear=True)
    mode_lines = [axs6[0].plot(np.nan, np.nan, color=f"C{s}")[0] for s in range(2)]
    meas_sc = axs6[0].scatter(np.nan, np.nan, color="r", marker="x")
    meas_sc_true = axs6[0].scatter(np.nan, np.nan, color="g", marker="x")
    min_ax = np.vstack(Z).min(axis=0)  # min(cell2mat(Z'));
    max_ax = np.vstack(Z).max(axis=0)  # max(cell2mat(Z'));
    axs6[0].axis([min_ax[0], max_ax[0], min_ax[1], max_ax[0]])

    for k, (Zk, pred_k, upd_k) in enumerate(
        zip(
            Z[plot_range],
            predict_list[plot_range],
            update_list[plot_range],
            # true_association[plot_range],
        ),
        start_k,
    ):
        (ax.cla() for ax in axs6)
        pl = []
        gated = tracker.gate(Zk, pred_k)  # probbar(:, k), xbar(:, :, k), Pbar(:, :, :, k));
        minG = 1e20 * np.ones(2)
        maxG = np.zeros(2)
        cond_upd_k = tracker.conditional_update(Zk[gated], pred_k)
        beta_k = tracker.association_probabilities(Zk[gated], pred_k)
        for s in range(2):
            mode_lines[s].set_data = (
                np.array([u.components[s].mean[:2] for u in update_list[:k]]).T,
            )
            axs6[1].plot(prob_hat[: (k - 1), s], color=f"C{s}")
            for j, cuj in enumerate(cond_upd_k):
                alpha = 0.7 * beta_k[j] * cuj.weights[s] + 0.3
                # csj = mTL * co(s, :) + (1 - mTL) * (beta(j)*skupd(s, j)*co(s, :) + (1 - beta(j)*skupd(s, j)) * ones(1, 3)); % transparancy
                upd_km1_s = update_list[k - 1].components[s]
                pl.append(
                    axs6[0].plot(
                        [upd_km1_s.mean[0], cuj.components[s].mean[0]],
                        [upd_km1_s.mean[1], cuj.components[s].mean[1]],
                        "--",
                        color=f"C{s}",
                        alpha=alpha,
                    )
                )

                pl.append(
                    axs6[1].plot(
                        [k - 1, k],
                        [prob_hat[k - 1, s], cuj.weights[s]],
                        color=f"C{s}",
                        alpha=alpha,
                    )
                )
                # axis([minAx(1), maxAx(1), minAx(2), maxAx(2)])
                #%alpha(pl, beta(j)*skupd(s, j));
                # drawnow;
                pl.append(
                    plot_cov_ellipse2d(
                        axs6[0],
                        cuj.components[s].mean[:2],
                        cuj.components[s].cov[:2, :2],
                        edgecolor=f"C{s}",
                        alpha=alpha,
                    )
                )

            Sk = tracker.state_filter.filters[s].innovation_cov([0, 0], pred_k.components[s])
            # gateData = chol(Sk)' * [cos(thetas); sin(thetas)] * sqrt(tracker.gateSize) + squeeze(xbar(1:2, s, k));
            # plot(gateData(1, :),gateData(2, :), '.--', 'Color', co(s,:))
            pl.append(
                plot_cov_ellipse2d(
                    axs6[0],
                    pred_k.components[s].mean[:2],
                    Sk,
                    n_sigma=tracker.gate_size,
                    edgecolor=f"C{s}",
                )
            )
            meas_sc.set_offsets(Zk)
            pl.append(axs6[0].scatter(*Zk.T, color="r", marker="x"))
            # if ak > 0:
            #     meas_sc_true.set_offsets(Zk[ak - 1])
            # else:
            #     meas_sc_true.set_offsets(np.array([np.nan, np.nan]))

            # for j = 1:size(xkupd, 3)
            #     csj = mTL * co(s, :) + (1 - mTL) * (beta(j)*skupd(s, j)*co(s, :) + (1 - beta(j)*skupd(s, j)) * ones(1, 3)); % transparancy
            #     plot([k-1, k], [probhat(s, k-1), skupd(s, j)], '--', 'color', csj)

            # minGs = min(gateData, [], 2);
            # minG = minGs .* (minGs < minG) + minG .* (minG < minGs);
            # maxGs = max(gateData, [], 2);
            # maxG = maxGs .* (maxGs > maxG) + maxG .* (maxG > maxGs);

        # scale = 1
        # minAx = minG - scale * (maxG - minG);
        # maxAx = maxG + scale * (maxG - minG);
        # axis([minAx(1), maxAx(1), minAx(2), maxAx(2)])
        # %legend()

        # mode probabilities

        # axis([1, plotRange(end), 0, 1])
        # drawnow;
        plt.pause(plot_pause)

    # %%

def plot_cov_ellipse2d(
    ax: plt.Axes,
    mean: np.ndarray = np.zeros(2),
    cov: np.ndarray = np.eye(2),
    n_sigma: float = 1,
    *,
    edgecolor: "Color" = "C0",
    facecolor: "Color" = "none",
    **kwargs,  # extra Ellipse keyword arguments
) -> matplotlib.patches.Ellipse:
    """Plot a n_sigma covariance ellipse centered in mean into ax."""
    ell_trans_mat = np.zeros((3, 3))
    ell_trans_mat[:2, :2] = np.linalg.cholesky(cov)
    ell_trans_mat[:2, 2] = mean
    ell_trans_mat[2, 2] = 1

    ell = matplotlib.patches.Ellipse(
        (0.0, 0.0),
        2.0 * n_sigma,
        2.0 * n_sigma,
        edgecolor=edgecolor,
        facecolor=facecolor,
        **kwargs,
    )
    trans = matplotlib.transforms.Affine2D(ell_trans_mat)
    ell.set_transform(trans + ax.transData)
    return ax.add_patch(ell)


