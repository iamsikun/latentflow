# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: cdivision=True
"""Cython-optimized routines for HMM EM algorithms."""

from libc.math cimport exp, log
import numpy as np
cimport numpy as np

ctypedef np.float64_t float64_t
ctypedef np.int64_t int64_t


cdef inline double _logsumexp_1d(double[::1] arr) nogil:
    cdef Py_ssize_t n = arr.shape[0]
    cdef Py_ssize_t i
    cdef double max_val = arr[0]
    cdef double val
    cdef double s = 0.0
    for i in range(1, n):
        val = arr[i]
        if val > max_val:
            max_val = val
    s = 0.0
    for i in range(n):
        val = arr[i]
        s += exp(val - max_val)
    return max_val + log(s)


def forward_log(object start_probs, object trans_mat, object log_b):
    """Forward algorithm in log-space using Cython loops."""

    cdef np.ndarray[np.float64_t, ndim=1] start_arr = np.ascontiguousarray(start_probs, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] trans_arr = np.ascontiguousarray(trans_mat, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] log_b_arr = np.ascontiguousarray(log_b, dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] log_start_arr = np.log(start_arr)
    cdef np.ndarray[np.float64_t, ndim=2] log_trans_arr = np.log(trans_arr)

    cdef Py_ssize_t T = log_b_arr.shape[0]
    cdef Py_ssize_t n_states = log_b_arr.shape[1]
    cdef Py_ssize_t t, i, j

    cdef np.ndarray[np.float64_t, ndim=2] log_alpha_arr = np.empty((T, n_states), dtype=np.float64)
    cdef double[:, ::1] log_alpha = log_alpha_arr
    cdef double[::1] log_start = log_start_arr
    cdef double[:, ::1] log_trans = log_trans_arr
    cdef double[:, ::1] log_b_view = log_b_arr

    cdef double max_val
    cdef double sum_val
    cdef double val

    for j in range(n_states):
        log_alpha[0, j] = log_start[j] + log_b_view[0, j]

    for t in range(1, T):
        for j in range(n_states):
            max_val = log_alpha[t - 1, 0] + log_trans[0, j]
            for i in range(1, n_states):
                val = log_alpha[t - 1, i] + log_trans[i, j]
                if val > max_val:
                    max_val = val
            sum_val = 0.0
            for i in range(n_states):
                val = log_alpha[t - 1, i] + log_trans[i, j]
                sum_val += exp(val - max_val)
            log_alpha[t, j] = max_val + log(sum_val) + log_b_view[t, j]

    cdef np.ndarray[np.float64_t, ndim=1] tmp_last = log_alpha_arr[-1]
    cdef double[::1] tmp_last_view = tmp_last
    cdef double loglik = _logsumexp_1d(tmp_last_view)

    return log_alpha_arr, float(loglik)


def backward_log(object trans_mat, object log_b):
    """Backward algorithm in log-space using Cython loops."""

    cdef np.ndarray[np.float64_t, ndim=2] trans_arr = np.ascontiguousarray(trans_mat, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] log_b_arr = np.ascontiguousarray(log_b, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] log_trans_arr = np.log(trans_arr)

    cdef Py_ssize_t T = log_b_arr.shape[0]
    cdef Py_ssize_t n_states = log_b_arr.shape[1]
    cdef Py_ssize_t t, i, j

    cdef np.ndarray[np.float64_t, ndim=2] log_beta_arr = np.zeros((T, n_states), dtype=np.float64)
    cdef double[:, ::1] log_beta = log_beta_arr
    cdef double[:, ::1] log_trans = log_trans_arr
    cdef double[:, ::1] log_b_view = log_b_arr

    cdef double max_val
    cdef double sum_val
    cdef double val

    for t in range(T - 2, -1, -1):
        for i in range(n_states):
            max_val = log_trans[i, 0] + log_b_view[t + 1, 0] + log_beta[t + 1, 0]
            for j in range(1, n_states):
                val = log_trans[i, j] + log_b_view[t + 1, j] + log_beta[t + 1, j]
                if val > max_val:
                    max_val = val
            sum_val = 0.0
            for j in range(n_states):
                val = log_trans[i, j] + log_b_view[t + 1, j] + log_beta[t + 1, j]
                sum_val += exp(val - max_val)
            log_beta[t, i] = max_val + log(sum_val)

    return log_beta_arr


def viterbi(object start_probs, object trans_mat, object log_b):
    """Viterbi decoding using Cython loops."""

    cdef np.ndarray[np.float64_t, ndim=1] start_arr = np.ascontiguousarray(start_probs, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] trans_arr = np.ascontiguousarray(trans_mat, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] log_b_arr = np.ascontiguousarray(log_b, dtype=np.float64)

    cdef np.ndarray[np.float64_t, ndim=1] log_start_arr = np.log(start_arr)
    cdef np.ndarray[np.float64_t, ndim=2] log_trans_arr = np.log(trans_arr)

    cdef Py_ssize_t T = log_b_arr.shape[0]
    cdef Py_ssize_t n_states = log_b_arr.shape[1]
    cdef Py_ssize_t t, i, j

    cdef np.ndarray[np.float64_t, ndim=2] delta_arr = np.empty((T, n_states), dtype=np.float64)
    cdef np.ndarray[np.int64_t, ndim=2] psi_arr = np.empty((T, n_states), dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] states_arr = np.empty(T, dtype=np.int64)

    cdef double[:, ::1] delta = delta_arr
    cdef long[:, ::1] psi = psi_arr
    cdef long[::1] states = states_arr
    cdef double[::1] log_start = log_start_arr
    cdef double[:, ::1] log_trans = log_trans_arr
    cdef double[:, ::1] log_b_view = log_b_arr

    cdef double best_val
    cdef long best_state
    cdef double val

    for j in range(n_states):
        delta[0, j] = log_start[j] + log_b_view[0, j]
        psi[0, j] = 0

    for t in range(1, T):
        for j in range(n_states):
            best_state = 0
            best_val = delta[t - 1, 0] + log_trans[0, j]
            for i in range(1, n_states):
                val = delta[t - 1, i] + log_trans[i, j]
                if val > best_val:
                    best_val = val
                    best_state = i
            delta[t, j] = best_val + log_b_view[t, j]
            psi[t, j] = best_state

    best_state = 0
    best_val = delta[T - 1, 0]
    for j in range(1, n_states):
        if delta[T - 1, j] > best_val:
            best_val = delta[T - 1, j]
            best_state = j
    states[T - 1] = best_state

    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]

    return states_arr
