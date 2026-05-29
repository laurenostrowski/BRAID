"""
Copyright (c) 2025 University of Southern California
See full notice in [LICENSE.md](LICENSE.md)
Parsa Vahidi, Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California
"""

"""The model used in BRAID"""
"""For mathematical descriptions see BRAIDModelDoc.md"""

import copy
import io
import logging
import os
import re
import time
import warnings
from datetime import datetime
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .MainModel import MainModel, shift_ms_to_1s_series
from .tools.abstract_classes import PredictorModel
from .tools.tf_losses import (
    masked_CategoricalCrossentropy,
    masked_CC,
    masked_mse,
    masked_PoissonLL_loss,
    masked_R2,
)
from .tools.tools import applyScaling, transposeIf, undoScaling

logger = logging.getLogger(__name__)

# --- safe helpers -----------------------------------
def _safe_transpose(data):
    if data is None: return None
    if isinstance(data, (list, tuple)): return [d.T for d in data]
    return data.T

def _safe_apply_scaling(sId, data, mean_name, std_name, missing_marker=None):
    if data is None: return None
    if isinstance(data, (list, tuple)):
        return [applyScaling(sId, d, mean_name, std_name, missing_marker=missing_marker) for d in data]
    return applyScaling(sId, data, mean_name, std_name, missing_marker=missing_marker)

def _safe_undo_scaling(sId, data, mean_name, std_name, missing_marker=None):
    if data is None: return None
    if isinstance(data, (list, tuple)):
        if len(data) > 0 and isinstance(data[0], (list, tuple)):
            return [[undoScaling(sId, step, mean_name, std_name, missing_marker=missing_marker) for step in trial] for trial in data]
        return [undoScaling(sId, d, mean_name, std_name, missing_marker=missing_marker) for d in data]
    return undoScaling(sId, data, mean_name, std_name, missing_marker=missing_marker)

def _is_nan_marker(missing_marker):
    return (missing_marker is not None
            and isinstance(missing_marker, float)
            and np.isnan(missing_marker))

def _clean_array(arr, missing_marker):
    """Replace missing values with 0.0. Returns a fresh array."""
    if missing_marker is None:
        return np.asarray(arr).copy()
    arr = np.asarray(arr)
    if _is_nan_marker(missing_marker):
        return np.where(np.isfinite(arr), arr, 0.0).astype(arr.dtype, copy=False)
    return np.where(arr == missing_marker, 0.0, arr).astype(arr.dtype, copy=False)

def _clean_signal_list(X, missing_marker):
    """Apply NaN/marker -> 0 cleaning to a (dim, T) array or list of such arrays."""
    if X is None:
        return None
    if isinstance(X, (list, tuple)):
        return [_clean_array(x, missing_marker) for x in X]
    return _clean_array(X, missing_marker)

def _count_nans(X, missing_marker):
    """Count NaN/marker entries in (dim, T) array or list of such arrays."""
    if X is None:
        return 0, 0
    def _arr_n(a):
        a = np.asarray(a)
        if missing_marker is None:
            return 0, a.size
        if _is_nan_marker(missing_marker):
            return int(np.sum(~np.isfinite(a))), a.size
        return int(np.sum(a == missing_marker)), a.size
    if isinstance(X, (list, tuple)):
        total = 0; n_nan = 0
        for x in X:
            n, t = _arr_n(x); n_nan += n; total += t
        return n_nan, total
    return _arr_n(X)
# ----------------------------------------------------

class BRAIDModel(MainModel):
    """The main class implementing BRAID. 
    x1(k+1) = A1( x1(k) ) + K1( y(k), u(k) )
    x2(k+1) = A2( x2(k) ) + K2( x1(k+1), y(k), u(k) )
    x3(k+1) = A3( x3(k) ) + K3( u(k) )
    y(k)    = Cy( x1(k), x2(k), u(k) ) + ey_k
    z(k)    = Cz( x1(k), x2(k), x3(k), u(k) ) + ez_k
    """
    def __init__(self, 
            block_samples=128,   
            batch_size=32,       
            log_dir = '', 
            missing_marker = None, 
            **kwargs):             
        
        self.block_samples = block_samples
        self.batch_size = batch_size
        self.log_dir = log_dir
        self.missing_marker = missing_marker

        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, Y, Z=None, U=None, nx=None, n1=None, n3=None, n_pre=None, 
            create_val_from_training = False, 
            validation_set_ratio = 0.2, 
            Y_validation = None, 
            Z_validation = None, 
            U_validation = None, 
            true_model = None,
            YType=None, ZType=None, UType=None,
            steps_ahead=None, 
            A_args = {}, K_args = {}, Cy_args = {}, Cz_args = {},    
            A1_args = None, K1_args = None, Cy1_args = None, Cz1_args = None, 
            A2_args = None, K2_args = None, Cy2_args = None, Cz2_args = None, 
            A3_args = None, K3_args = None, Cy3_args = None, Cz3_args = None, 
            args_base = None,
            noFT = False,
            noUZ = False,  
            isFullyLinear=False
        ): 

        if args_base is None:
            raise Exception("args_base should be provided externally to BRAIDModel fitting. It was None!")
        self.steps_ahead = args_base['steps_ahead']
        self.noFT = noFT
        self.noUZ = noUZ

        if isinstance(Y, (list, tuple)):
            ny, Ndat = Y[0].shape[0], sum(y.shape[1] for y in Y)
        else:
            ny, Ndat = Y.shape[0] if Y is not None else 0, Y.shape[1] if Y is not None else 0
            
        if Z is not None:
            if isinstance(Z, (list, tuple)):
                nz, NdatZ = Z[0].shape[0], sum(z.shape[1] for z in Z)
            else:
                nz, NdatZ = Z.shape[0], Z.shape[1]
        else:
            nz = 0
            
        if U is not None:
            nu = U[0].shape[0] if isinstance(U, (list, tuple)) else U.shape[0]
        else:
            nu = 0

        # ---------- Clean NaN in U before any BRAID processing ----------
        # BRAID's `RegressionModel.apply_func` does NOT handle NaN in its main
        # input (only in `prior_pred`). If U has NaN at any position, the Cy/Cz
        # regressions blow up immediately on the first gradient step:
        # dense(u_NaN) → NaN → loss=NaN → grad=NaN → weights=NaN.
        # The patched RNN cell handles NaN in U when U enters via `ny_in`, but
        # it doesn't help the downstream Cy/Cz regressions.
        #
        # Clean U to 0 at missing positions. After BRAID's z-scoring, those
        # positions will sit at (0 - mean) / std — a specific finite value,
        # consistent across train/val.
        if self.missing_marker is not None and U is not None:
            n_nan_u, total_u = _count_nans(U, self.missing_marker)
            if n_nan_u > 0:
                logger.info(
                    f"Cleaning {n_nan_u} NaN entries in U ({100*n_nan_u/total_u:.1f}%) "
                    f"-> 0 before BRAID training: BRAID's Cy/Cz regressions feed U "
                    f"directly into Dense layers without input masking."
                )
            U = _clean_signal_list(U, self.missing_marker)
            if U_validation is not None:
                U_validation = _clean_signal_list(U_validation, self.missing_marker)

        if n_pre is not None and n_pre > 0:
            args_pre = copy.deepcopy(args_base)
            args_pre['model2_Cz_Full']=False
            args_pre['allow_nonzero_Cz2']=True
            args_pre['has_UFT_reg']=False
            if noFT:
                args_pre['has_UFT']=False
            if noUZ:
                args_pre['has_UFT_z']=False
                args_pre['has_UFT_reg_z']=False
            sId_pre = MainModel(log_dir=self.log_dir, missing_marker=self.missing_marker)
            sId_pre.fit(Y, Z, U=U, nx=n_pre, n1=0,
                    YType=YType, ZType=ZType, 
                    Y_validation=Y_validation, Z_validation=Z_validation, U_validation=U_validation,
                    **args_pre)
        
            zPredRes1Train, _, _, _, _, _ = runPredict(sId_pre, Y=_safe_transpose(Y), Z=_safe_transpose(Z), U=_safe_transpose(U), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            if Y_validation is not None:
                zPredRes1Val, _, _, _, _, _ = runPredict(sId_pre, Y=_safe_transpose(Y_validation), Z=_safe_transpose(Z_validation), U=_safe_transpose(U_validation), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            else:
                zPredRes1Val = None

            if isinstance(Y, (list, tuple)):
                zPredRes1Train = [tp[0] for tp in zPredRes1Train]
                if zPredRes1Val is not None:
                    zPredRes1Val = [tp[0] for tp in zPredRes1Val]
            else:
                if isinstance(zPredRes1Train, list): zPredRes1Train = zPredRes1Train[0]
                if isinstance(zPredRes1Val, list): zPredRes1Val = zPredRes1Val[0]
            
        else:
            sId_pre = None
            zPredRes1Train = _safe_transpose(Z)
            zPredRes1Val = _safe_transpose(Z_validation)

        if n3 is None:
            n3 = 0
        if nx > n1+n3:
            nxThis = nx - n3; n1This = n1; n3This = n3; n2This = nx - n1This - n3This
        elif nx > n1:
            nxThis = n1; n1This = n1; n3This = nx - n1; n2This = 0
        else:
            nxThis = nx; n1This = nx; n3This = 0; n2This = 0

        args = copy.deepcopy(args_base)
        args['model2_Cz_Full']=False
        args['allow_nonzero_Cz2']=False 
        if noFT:
            args['has_UFT']=False
            args['has_UFT_reg']=False
        if noUZ: 
            args['has_UFT_z']=False
            args['has_UFT_reg_z']=False
            
        sId = MainModel(log_dir=self.log_dir, missing_marker=self.missing_marker)
        sId.fit(Y, _safe_transpose(zPredRes1Train), U=U, nx=nxThis, n1=n1This, 
                YType=YType, ZType=ZType, 
                Y_validation=Y_validation, Z_validation=_safe_transpose(zPredRes1Val), U_validation=U_validation,
                true_model=true_model, 
                **args)

        if n3This>0 and nu>0:
            args_post = copy.deepcopy(args_base)
            args_post['skip_Cy']=True
            args_post['allow_nonzero_Cz2']=False
            args_post['model2_Cz_Full']=False
            args_post['remove_flat_dims']=False
            if noFT:
                args_post['has_UFT']=False
                args_post['has_UFT_reg']=False
            if noUZ:
                args_post['has_UFT_z']=False
                args_post['has_UFT_reg_z']=False
            
            if A3_args is not None: 
                args_post['A1_args'] = copy.deepcopy(A3_args)
                if 'unifiedAK' in args_post['A1_args']: del args_post['A1_args']['unifiedAK']
            if K3_args is not None: 
                args_post['K1_args'] = copy.deepcopy(K3_args)
                if 'unifiedAK' not in args_post['K1_args']: args_post['K1_args']['unifiedAK'] = False
            if Cy3_args is not None: 
                args_post['Cy1_args'] = copy.deepcopy(Cy3_args)
                if 'unifiedAK' in args_post['Cy1_args']: del args_post['Cy1_args']['unifiedAK']
            if Cz3_args is not None: 
                args_post['Cz1_args'] = copy.deepcopy(Cz3_args)
                if 'unifiedAK' in args_post['Cz1_args']: del args_post['Cz1_args']['unifiedAK']

            zPredRes2Train, _, _, _, _, _ = runPredict(sId, Y=_safe_transpose(Y), Z=zPredRes1Train, U=_safe_transpose(U), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            if U_validation is not None:                
                zPredRes2Val, _, _, _, _, _ = runPredict(sId, Y=_safe_transpose(Y_validation), Z=zPredRes1Val, U=_safe_transpose(U_validation), YType=YType, ZType=ZType, useXFilt=False, missing_marker=self.missing_marker)
            else:
                zPredRes2Val = None

            if isinstance(Y, (list, tuple)):
                zPredRes2Train = [tp[0] for tp in zPredRes2Train]
                if zPredRes2Val is not None:
                    zPredRes2Val = [tp[0] for tp in zPredRes2Val]
            else:
                if isinstance(zPredRes2Train, list): zPredRes2Train = zPredRes2Train[0]
                if isinstance(zPredRes2Val, list): zPredRes2Val = zPredRes2Val[0]

            z2_train_T = _safe_transpose(zPredRes2Train)
            if isinstance(Z, (list, tuple)):
                Z3 = [z - zp for z, zp in zip(Z, z2_train_T)]
                if Z_validation is not None:
                    z2_val_T = _safe_transpose(zPredRes2Val)
                    Z3_validation = [z - zp for z, zp in zip(Z_validation, z2_val_T)]
                else:
                    Z3_validation = None
            else:
                Z3 = Z - z2_train_T
                if Z_validation is not None:
                    Z3_validation = Z_validation - _safe_transpose(zPredRes2Val)  
                else:
                    Z3_validation = None

            sId_post = MainModel(log_dir=self.log_dir, missing_marker=self.missing_marker)
            sId_post.fit(None, Z3, U=U, nx=n3This, n1=n3This, 
                    YType=YType, ZType=ZType, 
                    Y_validation=None, Z_validation=Z3_validation, U_validation=U_validation,
                    **args_post)
        else:
            sId_post = None

        self.sId_pre = sId_pre
        self.sId = sId
        self.sId_post = sId_post
        self.nx = nx
        self.n1 = n1This
        self.n3 = n3This
        self.n2 = n2This
        self.ny = ny
        self.nz = nz
        self.nu = nu
        self.blown_up = self.hasBlownUp()

    def hasBlownUp(self):
        blown_up = False
        if hasattr(self, 'sId') and  hasattr(self.sId, 'blown_up') and self.sId.blown_up:
            blown_up = True
        if hasattr(self, 'sId_pre') and  hasattr(self.sId_pre, 'blown_up') and self.sId_pre.blown_up:
            blown_up = True
        if hasattr(self, 'sId_post') and  hasattr(self.sId_post, 'blown_up') and self.sId_post.blown_up:
            blown_up = True
        return blown_up

    def discardModels(self):
        if hasattr(self, 'sId_pre') and hasattr(self.sId_pre, 'discardModels'):
            self.sId_pre.discardModels()
        if hasattr(self, 'sId') and hasattr(self.sId, 'discardModels'):
            self.sId.discardModels()
        if hasattr(self, 'sId_post') and hasattr(self.sId_post, 'discardModels'):
            self.sId_post.discardModels()

    def restoreModels(self):
        if hasattr(self, 'sId_pre') and hasattr(self.sId_pre, 'restoreModels'):
            self.sId_pre.restoreModels()
        if hasattr(self, 'sId') and hasattr(self.sId, 'restoreModels'):
            self.sId.restoreModels()
        if hasattr(self, 'sId_post') and hasattr(self.sId_post, 'restoreModels'):
            self.sId_post.restoreModels()

    def getLSSM(self): 
        return self.sId.getLSSM()

    def set_multi_step_with_data_gen(self, multi_step_with_data_gen, update_rnn_model_steps=True, noise_samples=0):
        self.sId.set_multi_step_with_data_gen(multi_step_with_data_gen, update_rnn_model_steps=update_rnn_model_steps, noise_samples=noise_samples)
        if self.n3 > 0 and hasattr(self, 'sId_post'):
            self.sId_post.set_multi_step_with_data_gen(multi_step_with_data_gen, update_rnn_model_steps=update_rnn_model_steps, noise_samples=noise_samples)

    def set_steps_ahead(self, steps_ahead, update_rnn_model_steps=True):
        self.sId.set_steps_ahead(steps_ahead, update_rnn_model_steps=update_rnn_model_steps)
        if self.n3 > 0 and hasattr(self, 'sId_post'):
            self.sId_post.set_steps_ahead(steps_ahead, update_rnn_model_steps=update_rnn_model_steps)
        if hasattr(self, 'steps_ahead'):
            self.steps_ahead = steps_ahead

    def predict(self, Y, U=None, x0=None):
        """Native prediction with safe fallback for Y=None and block patching"""
        is_list = isinstance(Y, (list, tuple)) or isinstance(U, (list, tuple))
        
        if is_list:
            lengths = [y.shape[0] for y in Y] if Y is not None else [u.shape[0] for u in U]
            Y_cat = np.concatenate(Y, axis=0) if Y is not None else None
            U_cat = np.concatenate(U, axis=0) if U is not None else None
            Ndat = sum(lengths)
            # Patch block samples using Total_Time to ensure unpadded trials process safely
            T_patch = Ndat 
            split_indices = np.cumsum(lengths)[:-1]
        else:
            Y_cat, U_cat = Y, U
            Ndat = Y.shape[0] if Y is not None else U.shape[0]
            T_patch = Ndat

        steps_ahead = self.steps_ahead if hasattr(self, 'steps_ahead') and self.steps_ahead is not None else [1]
        steps_ahead, _, steps_ahead_model1, _, model1_orig_step_inds \
             = self.sId.get_model_steps_ahead(steps_ahead)
             
        allXp_steps_cat = [np.zeros((Ndat, self.nx)) for s in steps_ahead]
        additionalArgs = {}
        
        # Patch block samples safely if modulo check fails
        patched = False
        bs = 1
        m_check = self.sId if hasattr(self, 'sId') else self
        
        # Safely extract block_samples, falling back to 1 if it's a linear model
        if hasattr(m_check, 'model1') and m_check.model1 is not None and hasattr(m_check.model1, 'block_samples'):
            bs = m_check.model1.block_samples
        elif hasattr(m_check, 'model2') and m_check.model2 is not None and hasattr(m_check.model2, 'block_samples'):
            bs = m_check.model2.block_samples

        if T_patch % bs != 0:
            saved_bs = _patch_block_samples(self, T_patch)
            patched = True
            
        try:
            preds = self.sId.predict(Y_cat, U=U_cat, **additionalArgs)
            allZp_steps_cat = list(preds[                  :  len(steps_ahead)])
            allYp_steps_cat = list(preds[  len(steps_ahead):2*len(steps_ahead)])
            allXp12_steps_cat = list(preds[2*len(steps_ahead):3*len(steps_ahead)])

            for saInd in range(len(steps_ahead)):
                allXp_steps_cat[saInd][:, :self.n1+self.n2] = allXp12_steps_cat[saInd]

            if self.n3 > 0 and hasattr(self, 'sId_post') and self.sId_post is not None:
                preds_post = self.sId_post.predict(None, U=U_cat, **additionalArgs)
                allZp3_steps_cat = preds_post[                  :  len(steps_ahead)]
                allXp3_steps_cat = preds_post[2*len(steps_ahead):3*len(steps_ahead)]
                for saInd in range(len(steps_ahead)):
                    allXp_steps_cat[saInd][:, self.n1+self.n2:] = allXp3_steps_cat[saInd]
                    allZp_steps_cat[saInd] = allZp_steps_cat[saInd] + allZp3_steps_cat[saInd]
        finally:
            if patched:
                _restore_block_samples(saved_bs)

        if is_list:
            allZp_split = [np.split(z, split_indices, axis=0) for z in allZp_steps_cat]
            allYp_split = [np.split(y, split_indices, axis=0) for y in allYp_steps_cat]
            allXp_split = [np.split(x, split_indices, axis=0) for x in allXp_steps_cat]
            allZp_steps = [allZp_split[saInd] for saInd in range(len(steps_ahead))]
            allYp_steps = [allYp_split[saInd] for saInd in range(len(steps_ahead))]
            allXp_steps = [allXp_split[saInd] for saInd in range(len(steps_ahead))]
        else:
            allZp_steps = allZp_steps_cat
            allYp_steps = allYp_steps_cat
            allXp_steps = allXp_steps_cat

        return tuple(allZp_steps) + tuple(allYp_steps) + tuple(allXp_steps)

def runPredWithArgs(args):
    return args[0].predict(args[1], args[2], **args[3])

def _patch_block_samples(model, T_total):
    saved = {}
    def _patch(m):
        if hasattr(m, 'model1') and m.model1 is not None and hasattr(m.model1, 'block_samples'):
            saved[m.model1] = m.model1.block_samples
            m.model1.block_samples = T_total
        if hasattr(m, 'model2') and m.model2 is not None and hasattr(m.model2, 'block_samples'):
            saved[m.model2] = m.model2.block_samples
            m.model2.block_samples = T_total

    if hasattr(model, 'sId'): 
        if getattr(model, 'sId_pre', None): _patch(model.sId_pre)
        if getattr(model, 'sId', None): _patch(model.sId)
        if getattr(model, 'sId_post', None): _patch(model.sId_post)
    else: 
        _patch(model)
    return saved

def _restore_block_samples(saved_dict):
    for rnn, old_bs in saved_dict.items():
        rnn.block_samples = old_bs

def runPredict(sId, Y=None, Z=None, U=None, YType=None, ZType=None, useXFilt=False, missing_marker=None, undo_scaling=False):
    """Runs the model prediction with fast batched processing and NaN-safe block patching."""    
    if YType == 'cont':
        Y = _safe_apply_scaling(sId, Y, 'yMean', 'yStd', missing_marker=missing_marker)
    if ZType == 'cont' and Z is not None:
        Z = _safe_apply_scaling(sId, Z, 'zMean', 'zStd', missing_marker=missing_marker)
    if U is not None:
        U = _safe_apply_scaling(sId, U, 'uMean', 'uStd', missing_marker=missing_marker)

    steps_ahead = [1]
    if hasattr(sId, 'steps_ahead') and sId.steps_ahead is not None:
        steps_ahead = sId.steps_ahead

    additionalArgs = {}
    if 'PSID.LSSM.LSSM' in str(type(sId)):
        additionalArgs['useXFilt'] = useXFilt
        
    is_list = isinstance(Y, (list, tuple)) or isinstance(U, (list, tuple))
    
    if is_list:
        n_trials = len(Y) if Y is not None else len(U)
        lengths = [y.shape[0] for y in Y] if Y is not None else [u.shape[0] for u in U]
        split_indices = np.cumsum(lengths)[:-1]
        
        # Patch dynamically using Total_Time to prevent modulo exceptions on unpadded data
        Total_Time = sum(lengths) 

        Y_cat = np.concatenate(Y, axis=0) if Y is not None else None
        U_cat = np.concatenate(U, axis=0) if U is not None else None
        
        saved_bs = _patch_block_samples(sId, Total_Time)
        try:
            predsThis = sId.predict(Y_cat, U=U_cat, **additionalArgs)
        finally:
            _restore_block_samples(saved_bs)

        zPred_cat = list(shift_ms_to_1s_series( predsThis[                  :  len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        yPred_cat = list(shift_ms_to_1s_series( predsThis[  len(steps_ahead):2*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        xPred_cat = list(shift_ms_to_1s_series( predsThis[2*len(steps_ahead):3*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        
        zPred_split = [np.split(z, split_indices, axis=0) for z in zPred_cat]
        yPred_split = [np.split(y, split_indices, axis=0) for y in yPred_cat]
        xPred_split = [np.split(x, split_indices, axis=0) for x in xPred_cat]
        
        zPred = [[zPred_split[step][trial] for step in range(len(steps_ahead))] for trial in range(n_trials)]
        yPred = [[yPred_split[step][trial] for step in range(len(steps_ahead))] for trial in range(n_trials)]
        xPred = [[xPred_split[step][trial] for step in range(len(steps_ahead))] for trial in range(n_trials)]

        if hasattr(sId, 'zErrSys') and hasattr(sId.zErrSys, 'UInEps') and sId.zErrSys.UInEps:
            sId.zErrSys.K = 0 * sId.zErrSys.K 
            sId.zErrSys.B_KD = sId.zErrSys.B
            sId.zErrSys.A_KC = sId.zErrSys.A
            
            saved_bs_eps = _patch_block_samples(sId.zErrSys, Total_Time)
            try:
                preds_eps = sId.zErrSys.predict(Y_cat, U=U_cat, **additionalArgs)
            finally:
                _restore_block_samples(saved_bs_eps)
                
            zPred_eps_cat = list(shift_ms_to_1s_series(preds_eps[:len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True))
            zPred_eps_split = [np.split(z, split_indices, axis=0) for z in zPred_eps_cat]
            
            for step in range(len(steps_ahead)):
                for trial in range(n_trials):
                    zPred[trial][step] += zPred_eps_split[step][trial]
        
    else:
        T_len = Y.shape[0] if Y is not None else U.shape[0]
        saved_bs = _patch_block_samples(sId, T_len)
        try:
            preds = sId.predict(Y, U, **additionalArgs)
        finally:
            _restore_block_samples(saved_bs)
            
        zPred = list(shift_ms_to_1s_series( preds[                  :  len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        yPred = list(shift_ms_to_1s_series( preds[  len(steps_ahead):2*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))
        xPred = list(shift_ms_to_1s_series( preds[2*len(steps_ahead):3*len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True ))

        if hasattr(sId, 'zErrSys') and hasattr(sId.zErrSys, 'UInEps') and sId.zErrSys.UInEps:
            sId.zErrSys.K = 0 * sId.zErrSys.K 
            sId.zErrSys.B_KD = sId.zErrSys.B
            sId.zErrSys.A_KC = sId.zErrSys.A
            
            saved_bs_eps = _patch_block_samples(sId.zErrSys, T_len)
            try:
                preds_eps = sId.zErrSys.predict(Y, U=U, **additionalArgs)
            finally:
                _restore_block_samples(saved_bs_eps)
                
            zPred_eps = list(shift_ms_to_1s_series(preds_eps[:len(steps_ahead)], steps_ahead, missing_marker=missing_marker, time_first=True))
            for z_sta_ind, zPred_step in enumerate(zPred):
                zPred[z_sta_ind] += zPred_eps[z_sta_ind]

    if undo_scaling:
        if YType == 'cont':
            yPred = _safe_undo_scaling(sId, yPred, 'yMean', 'yStd', missing_marker=missing_marker)
            Y = _safe_undo_scaling(sId, Y, 'yMean', 'yStd', missing_marker=missing_marker)
        if ZType == 'cont':
            zPred = _safe_undo_scaling(sId, zPred, 'zMean', 'zStd', missing_marker=missing_marker)
            Z = _safe_undo_scaling(sId, Z, 'zMean', 'zStd', missing_marker=missing_marker)
        if U is not None:
            U = _safe_undo_scaling(sId, U, 'uMean', 'uStd', missing_marker=missing_marker)
            
    return zPred, yPred, xPred, Y, Z, U