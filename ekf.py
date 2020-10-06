"""
Notation:
----------
x is generally used for either the state or the mean of a gaussian. It should be clear from context which it is.
P is used about the state covariance
z is a single measurement
Z (capital) are mulitple measurements so that z = Z[k] at a given time step
v is the innovation z - h(x)
S is the innovation covariance
"""
# %% Imports
# types
from typing import Union, Any, Dict, Optional, List, Sequence, Tuple, Iterable
from typing_extensions import Final

# packages
from dataclasses import dataclass, field
import numpy as np
import scipy.linalg as la
import scipy
from singledispatchmethod import singledispatchmethod

# local
import dynamicmodels as dynmods
import measurementmodels as measmods
from gaussparams import GaussParams, GaussParamList
from mixturedata import MixtureParameters

# %% The EKF


@dataclass
class EKF:
    # A Protocol so duck typing can be used
    dynamic_model: dynmods.DynamicModel
    # A Protocol so duck typing can be used
    sensor_model: measmods.MeasurementModel

    #_MLOG2PIby2: float = field(init=False, repr=False)

    @singledispatchmethod
    def init_filter_state(self, init) -> None:
        raise NotImplementedError(
            f"EKF do not know how to make {init} into GaussParams"
        )

    @init_filter_state.register(GaussParams)
    def _(self, init: GaussParams) -> GaussParams:
        return init

    @init_filter_state.register(tuple)
    @init_filter_state.register(list)
    def _(self, init: Union[Tuple, List]) -> GaussParams:
        return GaussParams(*init)

    @init_filter_state.register(dict)
    def _(self, init: dict) -> GaussParams:
        got_mean = False
        got_cov = False

        for key in init:
            if not got_mean and key in ["mean", "x", "m"]:
                mean = init[key]
                got_mean = True
            if not got_cov and key in ["cov", "P"]:
                cov = init[key]
                got_cov = True

        assert (
            got_mean and got_cov
        ), f"EKF do not recognize mean and cov keys in the dict {init}."

        return GaussParams(mean, cov)

    def __post_init__(self) -> None:
        self._MLOG2PIby2: Final[float] = self.sensor_model.m * \
            np.log(2 * np.pi) / 2

    def predict(self,
                ekfstate: GaussParams,
                # The sampling time in units specified by dynamic_model
                Ts: float,
                ) -> GaussParams:
        """Predict the EKF state Ts seconds ahead."""

        x, P = ekfstate  # tuple unpacking

        F = self.dynamic_model.F(x, Ts)
        Q = self.dynamic_model.Q(x, Ts)

        x_pred = F @ x
        P_pred = F @ P @ F.T + Q

        state_pred = GaussParams(x_pred, P_pred)

        return state_pred

    def innovation_mean(
            self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
    ) -> np.ndarray:
        """Calculate the innovation mean for ekfstate at z in sensor_state."""

        x = ekfstate.mean

        zbar = self.sensor_model.h(x)

        v = z - zbar

        return v

    def reduce_mixture(self,
            ekfstate_mixture: MixtureParameters[GaussParams]
            ) -> GaussParams:
        """ Merge a Gaussian mixture into a single mixture"""
        weights = ekfstate_mixture.weights
        means = np.array([component.mean for component in ekfstate_mixture.components])
        covs = np.array([component.cov for component in ekfstate_mixture.components])
        M = len(weights)
        d = len(means[0])

        mean = np.zeros_like(means[0])
        for i in range(M):
            mean += weights[i]*means[i]

        mean_diff = means - mean

        cov_mean = np.zeros_like(covs[0])
        cov_soi = np.zeros_like(covs[0])
        for i in range(M):
            cov_mean += weights[i] * covs[i]
            md = mean_diff[i][:,np.newaxis]
            cov_md = md @ md.T
            assert cov_md.shape == (d,d), f"wrong shape for {mean_diff}, expected {(M,M)}"
            cov_soi += weights[i] * cov_md

        cov = cov_mean + cov_soi

        return GaussParams(mean, cov)



    def innovation_cov(self,
                       z: np.ndarray,
                       ekfstate: GaussParams,
                       *,
                       sensor_state: Dict[str, Any] = None,
                       ) -> np.ndarray:
        """Calculate the innovation covariance for ekfstate at z in sensorstate."""

        x, P = ekfstate

        H = self.sensor_model.H(x, sensor_state=sensor_state)
        R = self.sensor_model.R(x, sensor_state=sensor_state, z=z)

        S = H @ P @ H.T + R

        return S

    def innovation(self,
                   z: np.ndarray,
                   ekfstate: GaussParams,
                   *,
                   sensor_state: Dict[str, Any] = None,
                   ) -> GaussParams:
        """Calculate the innovation for ekfstate at z in sensor_state."""

        # DONE: reuse the above functions for the innovation and its covariance
        v = self.innovation_mean(z, ekfstate)
        S = self.innovation_cov(z, ekfstate)

        innovationstate = GaussParams(v, S)

        return innovationstate

    def update(self,
               z: np.ndarray,
               ekfstate: GaussParams,
               sensor_state: Dict[str, Any] = None
               ) -> GaussParams:
        """Update ekfstate with z in sensor_state"""

        x, P = ekfstate

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        H = self.sensor_model.H(x, sensor_state=sensor_state)

        W = P @ H.T @ la.inv(S)

        x_upd = x + W @ v
        I = np.eye(x.shape[0])
        P_upd = (I - W @ H) @ P

        ekfstate_upd = GaussParams(x_upd, P_upd)

        return ekfstate_upd

    def step(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             # sampling time
             Ts: float,
             *,
             sensor_state: Dict[str, Any] = None,
             ) -> GaussParams:
        """Predict ekfstate Ts units ahead and then update this prediction with z in sensor_state."""

        # resue the above functions
        ekfstate_pred = self.predict(ekfstate, Ts)
        ekfstate_upd = self.update(z, ekfstate_pred)

        return ekfstate_upd

    def NIS(self,
            z: np.ndarray,
            ekfstate: GaussParams,
            *,
            sensor_state: Dict[str, Any] = None,
            ) -> float:
        """Calculate the normalized innovation squared for ekfstate at z in sensor_state"""

        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        NIS = v.T @ la.solve(S, v)

        return NIS

    @classmethod
    def NEES(cls,
             ekfstate: GaussParams,
             # The true state to comapare against
             x_true: np.ndarray,
             # NEES indices to use
             NEES_idx: Optional[Sequence[int]],
             ) -> float:
        """Calculate the normalized estimation error squared from ekfstate to x_true."""

        x, P = ekfstate

        x_diff = x - x_true
        x_diff = x_diff[NEES_idx]
        P_NEES = P[NEES_idx][:,NEES_idx]
        NEES = x_diff.T @ la.solve(P_NEES, x_diff)
        return NEES

    def gate(self,
             z: np.ndarray,
             ekfstate: GaussParams,
             gate_size_square: float,
             *,
             sensor_state: Dict[str, Any],
             ) -> bool:
        """ Check if z is inside sqrt(gate_sized_squared)-sigma ellipse of ekfstate in sensor_state """

        # a function to be used in PDA and IMM-PDA
        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)
        innov = v.T @ la.solve(S, v)

        return  innov < gate_size_square


    def loglikelihood(self,
                      z: np.ndarray,
                      ekfstate: GaussParams,
                      sensor_state: Dict[str, Any] = None
                      ) -> float:
        """Calculate the log likelihood of ekfstate at z in sensor_state"""
        # we need this function in IMM, PDA and IMM-PDA exercises
        # not necessary for tuning in EKF exercise
        v, S = self.innovation(z, ekfstate, sensor_state=sensor_state)

        # DONE: log likelihood, Hint: log(N(v, S))) -> NIS, la.slogdet.
        NIS = self.NIS(z, ekfstate)
        sign, logdet = np.linalg.slogdet(S)
        ll = -np.log(2*np.pi) - 0.5*logdet - 0.5*NIS

        return ll

    @classmethod
    def estimate(cls, ekfstate: GaussParams):
        """Get the estimate from the state with its covariance. (Compatibility method)"""
        # dummy function for compatibility with IMM class
        return ekfstate

    def estimate_sequence(
            self,
            # A sequence of measurements
            Z: Sequence[np.ndarray],
            # the initial KF state to use for either prediction or update (see start_with_prediction)
            init_ekfstate: GaussParams,
            # Time difference between Z's. If start_with_prediction: also diff before the first Z
            Ts: Union[float, Sequence[float]],
            *,
            # An optional sequence of the sensor states for when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # sets if Ts should be used for predicting before the first measurement in Z
            start_with_prediction: bool = False,
    ) -> Tuple[GaussParamList, GaussParamList]:
        """Create estimates for the whole time series of measurements."""

        # sequence length
        K = len(Z)

        # Create and amend the sampling array
        Ts_start_idx = int(not start_with_prediction)
        Ts_arr = np.empty(K)
        Ts_arr[Ts_start_idx:] = Ts
        # Insert a zero time prediction for no prediction equivalence
        if not start_with_prediction:
            Ts_arr[0] = 0

        # Make sure the sensor_state_list actually is a sequence
        sensor_state_seq = sensor_state or [None] * K

        # initialize and allocate
        ekfupd = init_ekfstate
        n = init_ekfstate.mean.shape[0]
        ekfpred_list = GaussParamList.allocate(K, n)
        ekfupd_list = GaussParamList.allocate(K, n)

        # perform the actual predict and update cycle
        # DONE loop over the data and get both the predicted and updated states in the lists
        # the predicted is good to have for evaluation purposes
        # A potential pythonic way of looping through  the data
        for k, (zk, Tsk, ssk) in enumerate(zip(Z, Ts_arr, sensor_state_seq)):
            if k == 0:
                ekfstate = ekfupd
            else:
                ekfstate = ekfupd_list[k-1]

            ekfpred_list[k] = self.predict(ekfstate, Tsk)
            ekfupd_list[k] = self.update(zk, ekfpred_list[k])

        return ekfpred_list, ekfupd_list

    def performance_stats(
            self,
            *,
            z: Optional[np.ndarray] = None,
            ekfstate_pred: Optional[GaussParams] = None,
            ekfstate_upd: Optional[GaussParams] = None,
            sensor_state: Optional[Dict[str, Any]] = None,
            x_true: Optional[np.ndarray] = None,
            # None: no norm, -1: all idx, seq: a single norm for given idxs, seqseq: a norms for idxseq
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs, see np.linalg.norm ord argument.
            norms: Union[Iterable[int], int] = 2,
            # NEES indices to use
            NEES_idx: Optional[Sequence[int]] = None,
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculate performance statistics available from the given parameters."""
        stats: Dict[str, Union[float, List[float]]] = {}

        # NIS, needs measurements
        if z is not None and ekfstate_pred is not None:
            stats['NIS'] = self.NIS(
                z, ekfstate_pred, sensor_state=sensor_state)

        # NEES and RMSE, needs ground truth
        if x_true is not None:
            # prediction
            if ekfstate_pred is not None:
                stats['NEESpred'] = self.NEES(ekfstate_pred, x_true, NEES_idx)

                # distances
                err_pred = ekfstate_pred.mean - x_true
                if norm_idxs is None:
                    stats['dist_pred'] = np.linalg.norm(err_pred, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_pred'] = [
                        np.linalg.norm(err_pred[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]

            # update
            if ekfstate_upd is not None:
                stats['NEESupd'] = self.NEES(ekfstate_upd, x_true, NEES_idx)

                # distances
                err_upd = ekfstate_upd.mean - x_true
                if norm_idxs is None:
                    stats['dist_upd'] = np.linalg.norm(err_upd, ord=norms)
                elif isinstance(norm_idxs, Iterable) and isinstance(norms, Iterable):
                    stats['dists_upd'] = [
                        np.linalg.norm(err_upd[idx], ord=ord)
                        for idx, ord in zip(norm_idxs, norms)]
        return stats

    def performance_stats_sequence(
            self,
            # Sequence length
            K: int,
            *,
            # The measurements
            Z: Optional[Iterable[np.ndarray]] = None,
            ekfpred_list: Optional[Iterable[GaussParams]] = None,
            ekfupd_list: Optional[Iterable[GaussParams]] = None,
            # An optional sequence of all the sensor states when Z was recorded
            sensor_state: Optional[Iterable[Optional[Dict[str, Any]]]] = None,
            # Optional ground truth for error checking
            X_true: Optional[Iterable[Optional[np.ndarray]]] = None,
            # Indexes to be used to calculate errornorms, multiple to separate the state space.
            # None: all idx, Iterable (eg. list): each element is an index sequence into the dimension of the state space.
            norm_idxs: Optional[Iterable[Sequence[int]]] = None,
            # The sequence of norms to calculate for idxs (see numpy.linalg.norm ord argument).
            norms: Union[Iterable[int], int] = 2,
            # NEES indices to use
            NEES_idx: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Get performance metrics on a pre-estimated sequence"""

        None_list = [None] * K

        for_iter = []
        for_iter.append(Z if Z is not None else None_list)
        for_iter.append(ekfpred_list or None_list)
        for_iter.append(ekfupd_list or None_list)
        for_iter.append(sensor_state or None_list)
        for_iter.append(X_true if X_true is not None else None_list)

        stats = []
        for zk, ekfpredk, ekfupdk, ssk, xtk in zip(*for_iter):
            stats.append(
                self.performance_stats(
                    z=zk, ekfstate_pred=ekfpredk, ekfstate_upd=ekfupdk, sensor_state=ssk, x_true=xtk,
                    norm_idxs=norm_idxs, norms=norms, NEES_idx=NEES_idx
                )
            )

        # make structured array
        dtype = [(key, *((type(val[0]), len(val)) if isinstance(val, Iterable)
                         else (type(val),))) for key, val in stats[0].items()]
        stats_arr = np.array([tuple(d.values()) for d in stats], dtype=dtype)

        return stats_arr


# %% End
