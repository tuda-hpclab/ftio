"""
Author: lucasch03
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: 0.0.8
Date: Jan 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

from argparse import Namespace

import numpy as np
import pytest

import ftio.prediction.unify_predictions as unify_predictions
from ftio.freq._analysis_figures import AnalysisFigures
from ftio.freq.prediction import Prediction
from ftio.prediction.unify_predictions import merge_core, merge_predictions

"""
Tests for class ftio/prediction/unify_predictions.py
"""


def test_merge_predictions():
    pred = Prediction(transformation="dft")
    pred.dominant_freq = np.array([0.1, 0.2, 0.3])
    pred.conf = np.array([0.7, 0.8, 0.6])
    pred.amp = np.array([1.0, 2.0, 0.5])
    pred.phi = np.array([0.0, 0.1, 0.2])

    pred2 = Prediction(transformation="autocorrelation")
    pred2.dominant_freq = np.array([0.2])
    pred2.conf = np.array([0.85])
    pred2.amp = np.array([1.5])
    pred2.phi = np.array([0.05])
    pred2.candidates = np.array([4.9, 5.0, 5.1])  # ~1/0.2 = 5 sec period

    merged, text = merge_core(pred, pred2, freq=10.0, text="")

    assert isinstance(merged, Prediction)
    assert isinstance(text, str)


def test_merge_predictions_keeps_amp_phi_periodicity_in_sync_with_dominant_freq():
    # merge_core narrows dominant_freq/conf down to the single merge winner but
    # used to leave amp/phi/periodicity at their pre-merge, longer length (still
    # indexed by the original DFT candidate array). get_dominant_index() then
    # argmaxes over the stale, longer amp array and can return an index valid
    # there but out of bounds for the now-shorter dominant_freq -> IndexError in
    # get_dominant_freq_and_conf(). Regression for that: with 3 DFT candidates
    # where the merge winner is not index 0, all four arrays must end up the
    # same length and get_dominant_freq_and_conf() must not raise.
    pred = Prediction(transformation="dft")
    pred.dominant_freq = np.array([0.1, 0.2, 0.3])
    pred.conf = np.array([0.7, 0.8, 0.6])
    pred.amp = np.array([1.0, 2.0, 0.5])
    pred.phi = np.array([0.0, 0.1, 0.2])
    pred.periodicity = np.array([0.7, 0.8, 0.6])

    pred2 = Prediction(transformation="autocorrelation")
    pred2.dominant_freq = np.array([0.2])
    pred2.conf = np.array([0.85])
    pred2.candidates = np.array([4.9, 5.0, 5.1])  # ~1/0.2 = 5 sec period

    merged, _ = merge_core(pred, pred2, freq=10.0, text="")

    assert len(merged.dominant_freq) == 1
    assert len(merged.amp) == len(merged.dominant_freq)
    assert len(merged.phi) == len(merged.dominant_freq)
    assert len(merged.periodicity) == len(merged.dominant_freq)
    assert merged.dominant_freq[0] == pytest.approx(0.2)
    assert merged.amp[0] == pytest.approx(2.0)  # amp at the winning index (1)

    freq, conf = merged.get_dominant_freq_and_conf()
    assert freq == pytest.approx(0.2)
    assert conf == pytest.approx(merged.conf[0])


def test_merge_predictions_survives_a_low_confidence_empty_merge_result(monkeypatch):
    # merge_predictions used to do `if pred_merged.dominant_freq:` -- a bare
    # truthiness check on a numpy array. merge_core's low-confidence branch
    # returns an *empty* dominant_freq (conf < 0.2, see merge_core), and
    # `bool(np.array([]))` raises ValueError ("truth value of an empty array is
    # ambiguous"), not False. Confirmed live on BSC (job 43609316): the online
    # predictor crashed here the first time it hit a low-confidence merge.
    empty_merged = Prediction(transformation="dft")
    empty_merged.dominant_freq = np.array([])
    empty_merged.conf = np.array([])

    def fake_merge_core(pred_dft, pred_auto, freq, text):
        return empty_merged, text

    monkeypatch.setattr(unify_predictions, "merge_core", fake_merge_core)

    pred_dft = Prediction(transformation="dft")
    pred_dft.dominant_freq = np.array([0.1, 0.2])
    pred_dft.conf = np.array([0.3, 0.3])

    pred_auto = Prediction(transformation="autocorrelation")
    pred_auto.dominant_freq = np.array([0.2])
    pred_auto.conf = np.array([0.3])

    args = Namespace(
        autocorrelation=True,
        transformation="dft",
        verbose=False,
        engine="no",
        freq=10.0,
    )

    merged = merge_predictions(args, pred_dft, pred_auto, AnalysisFigures())

    assert len(merged.dominant_freq) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
