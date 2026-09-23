"""CPU self-tests of the task sampler (depth curriculum and depth rebalancing).

In plain words: checks that depth rebalancing gives each depth the intended share of
draws, that the calendar depth curriculum only allows the right depths at each epoch,
that the automatic curriculum moves to the next depth when the training reward is
high enough, and that the weighted sampler yields batches with the same shape as
TRL's own sampler. Run:  python -m src.tests.test_sampling
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.train.sampling import (DepthAutoScheduleProvider, DepthScheduleProvider,  # noqa: E402
                                WeightedRepeatSampler, depth_balance_probs)


def main() -> None:
    depths = [1] * 109 + [2] * 239 + [3] * 88 + [4] * 8
    d = np.asarray(depths)
    p = depth_balance_probs(depths, "uniform")
    assert np.allclose([p[d == k].sum() for k in (1, 2, 3, 4)], 0.25) and np.isclose(p.sum(), 1)
    q = depth_balance_probs(depths, "sqrt")
    expected = np.sqrt([109, 239, 88, 8]); expected /= expected.sum()
    assert np.allclose([q[d == k].sum() for k in (1, 2, 3, 4)], expected)

    sch = DepthScheduleProvider("1:0,2:6,3:20,4:45", depths)
    sch.trainer_ref = SimpleNamespace(state=SimpleNamespace(epoch=7.0))
    assert set(d[sch.probabilities() > 0]) == {1, 2}

    auto = DepthAutoScheduleProvider(depths, steps_per_epoch=4, threshold=0.8, max_stage_epochs=10)
    for i in range(4):
        auto.on_log(None, None, None, logs={"reward": 0.9, "epoch": 0.25 * (i + 1)})
    assert auto.stage == 2, auto.stage
    assert set(d[auto.probabilities() > 0]) == {1, 2}

    s = WeightedRepeatSampler(list(range(len(depths))), mini_repeat_count=16, batch_size=4,
                              repeat_count=1, prob_fn=lambda: p, seed=0)
    idx = list(iter(s))
    assert len(idx) == len(s) and all(idx[i] == idx[i - i % 16] for i in range(len(idx)))
    print("[test_sampling] OK — rebalancing uniform/sqrt, calendar and automatic depth "
          "curricula, sampler shape identical to TRL's RepeatSampler")


if __name__ == "__main__":
    main()
