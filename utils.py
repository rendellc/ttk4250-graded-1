from dataclasses import dataclass

import estimationstatistics as estats

import numpy as np
from typing import List
from gaussparams import GaussParams

@dataclass
class TrackResult:
    """
    Dataclass for storing results from tracking on dataset.
    """
    x_hat: np.ndarray
    prob_hat: np.ndarray
    predict_list: List[GaussParams]
    update_list: List[GaussParams]
    estimate_list: List[GaussParams]
    NEES: List[float]
    NEESpos: List[float]
    NEESvel: List[float]
    ANEES: float
    ANEESpos: float
    ANEESvel: float
    pos_error: np.ndarray
    vel_error: np.ndarray
    posRMSE: float
    velRMSE: float
    peak_pos_deviation: float
    peak_vel_deviation: float




def compute_rmse_and_peak_deviation(v1, v2):
    err = np.linalg.norm(v1 - v2, axis=0)
    rmse = np.sqrt(np.mean(err**2))
    peak_deviation = err.max()
    return rmse, peak_deviation

def track_and_evaluate_sequence(tracker, init_state, Z, Xgt, Ts, K):
    NEES = np.zeros(K)
    NEESpos = np.zeros(K)
    NEESvel = np.zeros(K)

    update = init_state
    update_list = []
    predict_list = []
    estimate_list = []

    # estimate
    for k, (Zk, x_true_k) in enumerate(zip(Z, Xgt)):
        if isinstance(Ts, float):
            predict = tracker.predict(update, Ts)
        else:
            print("DEBUG: this should be some sort of list:", type(Ts))
            predict = tracker.predict(update, Ts[k])
        update = tracker.update(Zk, predict)

        # You can look at the prediction estimate as well
        estimate = tracker.estimate(update)

        NEES[k] = estats.NEES_indexed(
            estimate.mean, estimate.cov, x_true_k, idxs=np.arange(4)
        )

        NEESpos[k] = estats.NEES_indexed(
            estimate.mean, estimate.cov, x_true_k, idxs=np.arange(2)
        )
        NEESvel[k] = estats.NEES_indexed(
            estimate.mean, estimate.cov, x_true_k, idxs=np.arange(2, 4)
        )

        predict_list.append(predict)
        update_list.append(update)
        estimate_list.append(estimate)

    x_hat = np.array([est.mean for est in estimate_list])
    prob_hat = np.array([upd.weights for upd in update_list])
    pos_error = np.linalg.norm(x_hat[:, :2] - Xgt[:, :2], axis=1)
    vel_error = np.linalg.norm(x_hat[:, 2:4] - Xgt[:, 2:4], axis=1)
    posRMSE, peak_pos_deviation = compute_rmse_and_peak_deviation(x_hat[:, :2], Xgt[:, :2])
    velRMSE, peak_vel_deviation = compute_rmse_and_peak_deviation(x_hat[:, 2:4], Xgt[:, 2:4])

    assert len(pos_error) == K

    results = TrackResult(
        x_hat=x_hat,
        prob_hat=prob_hat,
        predict_list=predict_list,
        update_list=update_list,
        estimate_list=estimate_list,
        NEES=NEES,
        NEESpos=NEESpos,
        NEESvel=NEESvel,
        ANEES=np.mean(NEES),
        ANEESpos=np.mean(NEESpos),
        ANEESvel=np.mean(NEESvel),
        pos_error=pos_error,
        vel_error=vel_error,
        posRMSE=posRMSE,
        velRMSE=velRMSE,
        peak_pos_deviation=peak_pos_deviation,
        peak_vel_deviation=peak_vel_deviation)

    return results


def trajectory_plot(ax, trackresult, Xgt):
    tr = trackresult

    ax.plot(*tr.x_hat.T[:2], label=r"$\hat x$")
    ax.plot(*Xgt.T[:2], label="$x$")
    ax.set_title(
        f"RMSE(pos, vel) = ({tr.posRMSE:.3f}, {tr.velRMSE:.3f})\npeak_dev(pos, vel) = ({tr.peak_pos_deviation:.3f}, {tr.peak_vel_deviation:.3f})"
    )
    ax.axis("equal")

def mode_plot(ax, trackresult, K, Ts):
    assert isinstance(Ts, float), "need to create mode_plot for Ts: List[float] aswell"

    # probabilities
    ax.plot(np.arange(K) * Ts, trackresult.prob_hat)
    ax.set_ylim([0, 1])
    ax.set_ylabel("mode probability")
    ax.set_xlabel("time")

def confidence_interval_plot(ax, NEES, K, Ts, CI, confprob, ylabel):
    assert isinstance(Ts, float), "need to create function for Ts: List[float] aswell"

    inCI = np.mean((CI[0] <= NEES) * (NEES <= CI[1]))

    ax.plot(np.arange(K) * Ts, NEES)
    ax.plot([0, (K - 1) * Ts], np.repeat(CI[None], 2, 0), "--r")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{inCI*100:.1f}% inside {confprob*100:.1f}% CI")

