"""
Tensorflow losses with NaN-safe missing-data handling.

The standard masked-loss pattern here is:

  1. Detect missingness with a NaN-safe predicate.
     - If `missing_marker` is NaN, equality is useless (NaN != NaN is True by IEEE
       754), so we use `tf.math.is_finite` instead.
     - For any other (finite) marker, `tf.not_equal` is fine.
  2. Replace NaN/missing values in the target with a safe value (zero) BEFORE
     any arithmetic, so the loss expression never produces NaN.
  3. Multiply per-sample loss by a binary 0/1 mask. With both factors finite,
     the multiplication can't poison the result. Normalize by the mask sum.

This pattern is internally consistent: the same `_compute_mask_and_safe`
helper is used by every loss in this file, so flipping the marker semantics
in one place doesn't drift the others.
"""

import numpy as np
import tensorflow as tf


def _is_nan_marker(mask_value):
    """True iff mask_value is float NaN. Safe for None / int / np types."""
    if mask_value is None:
        return False
    try:
        return bool(np.isnan(mask_value))
    except (TypeError, ValueError):
        return False


def _compute_mask_and_safe(y_true, mask_value, sample_axes=(-1,)):
    """
    Compute (is_valid_per_element, is_valid_per_sample, y_true_safe).

    Args:
        y_true: tensor of any rank.
        mask_value: None, np.nan, or a finite scalar sentinel.
        sample_axes: axes over which to reduce to get per-sample validity.
            For typical (N, D) inputs this is (-1,); for (N, T, C) one-hot
            categorical it can be (-2, -1).

    Returns:
        isOk_elem (bool tensor, shape of y_true): per-element validity.
        mask_sample (y_true.dtype, sample reduction): 1.0 if ALL elements in the
            reduced axes are valid, else 0.0.
        y_true_safe (y_true.dtype, shape of y_true): y_true with invalid entries
            replaced by 0 so downstream arithmetic stays finite.
    """
    if mask_value is None:
        isOk_elem = tf.ones_like(y_true, dtype=tf.bool)
    elif _is_nan_marker(mask_value):
        # NaN-safe predicate: also treats +/- inf as invalid, which is what
        # we want — non-finite values would poison the loss regardless.
        isOk_elem = tf.math.is_finite(y_true)
    else:
        marker = tf.constant(mask_value, dtype=y_true.dtype)
        isOk_elem = tf.not_equal(y_true, marker)

    mask_sample = tf.cast(
        tf.reduce_all(isOk_elem, axis=list(sample_axes)),
        dtype=y_true.dtype,
    )
    y_true_safe = tf.where(isOk_elem, y_true, tf.zeros_like(y_true))
    return isOk_elem, mask_sample, y_true_safe


# -----------------------------------------------------------------------------
# Regression losses
# -----------------------------------------------------------------------------

def masked_mse(mask_value=None):
    """Mean squared error with NaN-safe per-sample masking.

    A sample is dropped from the average iff any of its target dims is missing.

    Note on defensive `tf.where` for the per-sample loss gate: we use
    `tf.where(mask>0, per_sample, 0)` rather than multiplying by the mask.
    Multiplication breaks if per_sample contains NaN at masked positions
    (e.g. when y_pred itself is NaN at those positions due to NaN inputs
    upstream), since `0 * NaN = NaN`. The `tf.where` form makes the forward
    pass robust to such upstream NaN at masked positions. Backward through
    `tf.where(c, x, y)` can still produce NaN gradients if x has NaN values —
    the only real cure for that is to prevent NaN from entering the model in
    the first place by cleaning inputs upstream.
    """
    def f(y_true, y_pred):
        sh = tf.shape(y_true)
        y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
        y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
        y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)

        _, mask_sample, y_true_safe = _compute_mask_and_safe(
            y_true_f, mask_value, sample_axes=(-1,))

        # MSE-per-sample on cleaned target (finite everywhere)
        sq_err = tf.square(y_pred_f - y_true_safe)
        per_sample = tf.reduce_mean(sq_err, axis=-1)
        # Use tf.where (not multiply): masked positions get exactly 0, even
        # if per_sample is NaN there (e.g. from NaN y_pred at masked positions).
        masked_loss = tf.where(
            mask_sample > 0,
            per_sample,
            tf.zeros_like(per_sample),
        )

        valid_count = tf.reduce_sum(mask_sample)
        return tf.reduce_sum(masked_loss) / (valid_count + 1e-8)

    f.__name__ = str("MSE_maskV_{}".format(mask_value))
    return f


def computeCC_masked(y_true, y_pred, mask_value=None):
    """Per-output-dim correlation coefficient, ignoring masked samples."""
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)

    _, mask_sample, y_true_safe = _compute_mask_and_safe(
        y_true_f, mask_value, sample_axes=(-1,))
    mask_col = tf.expand_dims(mask_sample, axis=-1)   # (N, 1) for broadcasting
    valid_count = tf.reduce_sum(mask_col, axis=0) + 1e-8

    # Per-dim mean on masked (clean) data
    mx = tf.reduce_sum(y_true_safe * mask_col, axis=0) / valid_count
    my = tf.reduce_sum(y_pred_f    * mask_col, axis=0) / valid_count

    # Centered deviations, with invalid samples zeroed out
    xm = (y_true_safe - mx) * mask_col
    ym = (y_pred_f    - my) * mask_col

    r_num = tf.reduce_sum(xm * ym, axis=0)
    r_den = tf.sqrt(tf.reduce_sum(tf.square(xm), axis=0)) \
          * tf.sqrt(tf.reduce_sum(tf.square(ym), axis=0))
    return r_num / (r_den + 1e-8)


def computeR2_masked(y_true, y_pred, mask_value=None):
    """Per-output-dim R^2, ignoring masked samples. Returns 0 for flat targets."""
    sh = tf.shape(y_true)
    y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-1]), sh[-1]])
    y_true_f = tf.cast(y_true_r, dtype=y_pred.dtype)
    y_pred_f = tf.cast(y_pred_r, dtype=y_pred.dtype)

    _, mask_sample, y_true_safe = _compute_mask_and_safe(
        y_true_f, mask_value, sample_axes=(-1,))
    mask_col = tf.expand_dims(mask_sample, axis=-1)
    valid_count = tf.reduce_sum(mask_col, axis=0) + 1e-8

    m_true = tf.reduce_sum(y_true_safe * mask_col, axis=0) / valid_count

    r_num = tf.reduce_sum(tf.square(y_true_safe - y_pred_f)  * mask_col, axis=0)
    r_den = tf.reduce_sum(tf.square(y_true_safe - m_true)    * mask_col, axis=0)
    R2 = 1.0 - (r_num / (r_den + 1e-8))

    # Detect dims where the (masked) target has effectively zero range,
    # report R^2 = 0 for them (otherwise R^2 is ill-defined and dominated by noise).
    # Using sentinels far outside the data range so masked positions don't perturb min/max.
    big = tf.cast(1e9, y_pred.dtype)
    max_y = tf.reduce_max(y_true_safe * mask_col + (1.0 - mask_col) * -big, axis=0)
    min_y = tf.reduce_min(y_true_safe * mask_col + (1.0 - mask_col) *  big, axis=0)
    isFlat = (max_y - min_y) < tf.cast(1e-9, y_pred.dtype)
    R2 = tf.where(isFlat, tf.zeros_like(R2), R2)
    return R2


def masked_CC(mask_value=None):
    def f(y_true, y_pred):
        return tf.math.reduce_mean(computeCC_masked(y_true, y_pred, mask_value))
    f.__name__ = str("CC_maskV_{}".format(mask_value))
    return f


def masked_R2(mask_value=None):
    def f(y_true, y_pred):
        return tf.math.reduce_mean(computeR2_masked(y_true, y_pred, mask_value))
    f.__name__ = str("R2_maskV_{}".format(mask_value))
    return f


def masked_negativeCC(mask_value=None):
    def f(y_true, y_pred):
        return -tf.math.reduce_mean(computeCC_masked(y_true, y_pred, mask_value))
    f.__name__ = str("negCC_maskV_{}".format(mask_value))
    return f


def masked_negativeR2(mask_value=None):
    def f(y_true, y_pred):
        return -tf.math.reduce_mean(computeR2_masked(y_true, y_pred, mask_value))
    f.__name__ = str("negR2_maskV_{}".format(mask_value))
    return f


# -----------------------------------------------------------------------------
# Poisson loss (count data)
# -----------------------------------------------------------------------------

def masked_PoissonLL_loss(mask_value=None):
    """Poisson NLL with NaN-safe masking. For integer count targets."""
    def f(true_counts, pred_logLambda):
        sh = tf.shape(true_counts)
        true_counts_f = tf.reshape(true_counts, [tf.reduce_prod(sh[:-1]), sh[-1]])
        pred_logLambda_f = tf.reshape(pred_logLambda, [tf.reduce_prod(sh[:-1]), sh[-1]])

        # Poisson markers are conventionally integer; honor that here for the
        # non-NaN branch by casting through float for the predicate but using
        # the original dtype for the safe value.
        _, mask_sample, true_counts_safe = _compute_mask_and_safe(
            true_counts_f, mask_value, sample_axes=(-1,))
        mask_f = tf.cast(mask_sample, pred_logLambda_f.dtype)

        lossFunc = tf.keras.losses.Poisson(reduction=tf.keras.losses.Reduction.NONE)
        raw_loss = lossFunc(true_counts_safe, pred_logLambda_f)
        # tf.where, not multiply: NaN-safe forward at masked positions.
        masked_loss = tf.where(mask_f > 0, raw_loss, tf.zeros_like(raw_loss))

        valid_count = tf.reduce_sum(mask_f)
        return tf.reduce_sum(masked_loss) / (valid_count + 1e-8)

    f.__name__ = str("PoissonLL_maskV_{}".format(mask_value))
    return f


# -----------------------------------------------------------------------------
# Categorical losses
# -----------------------------------------------------------------------------

def masked_CategoricalCrossentropy(mask_value=None):
    """Categorical CE on one-hot targets (..., n_outs, n_classes)."""
    def f(y_true, y_pred):
        sh = tf.shape(y_true)
        y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])
        y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh[:-2]), sh[-2], sh[-1]])

        # For CE, reduce over both the n_outs and n_classes axes to decide
        # "is this sample valid". A sample is invalid if ANY output dim or
        # ANY class label slot has the marker.
        _, mask_sample, y_true_safe = _compute_mask_and_safe(
            y_true_r, mask_value, sample_axes=(-2, -1))
        mask_f = tf.cast(mask_sample, y_pred_r.dtype)

        lossFunc = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        raw_loss = lossFunc(y_true_safe, y_pred_r)
        masked_loss = tf.where(mask_f > 0, raw_loss, tf.zeros_like(raw_loss))

        valid_count = tf.reduce_sum(mask_f)
        return tf.reduce_sum(masked_loss) / (valid_count + 1e-8)

    f.__name__ = str("CCE_maskV_{}".format(mask_value))
    return f


def masked_SparseCategoricalCrossentropy(mask_value=None):
    """Sparse categorical CE on integer-label targets (..., n_outs)."""
    def f(y_true, y_pred):
        sh = tf.shape(y_true)
        y_true_r = tf.reshape(y_true, [tf.reduce_prod(sh[:-1]), sh[-1]])

        sh2 = tf.shape(y_pred)
        y_pred_r = tf.reshape(y_pred, [tf.reduce_prod(sh2[:-2]), sh2[-2], sh2[-1]])

        _, mask_sample, y_true_safe = _compute_mask_and_safe(
            y_true_r, mask_value, sample_axes=(-1,))
        mask_f = tf.cast(mask_sample, y_pred_r.dtype)

        lossFunc = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        raw_loss = lossFunc(y_true_safe, y_pred_r)
        masked_loss = tf.where(mask_f > 0, raw_loss, tf.zeros_like(raw_loss))

        valid_count = tf.reduce_sum(mask_f)
        return tf.reduce_sum(masked_loss) / (valid_count + 1e-8)

    f.__name__ = str("SCCE_maskV_{}".format(mask_value))
    return f