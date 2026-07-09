"""
Author: lucasch03
Copyright (c) 2024-2026 TU Darmstadt, Germany
Version: 0.0.8
Date: Jan 2026

Licensed under the BSD 3-Clause License.
For more information, see the LICENSE file in the project root:
https://github.com/tuda-parallel/FTIO/blob/main/LICENSE
"""

import pytest

from ftio.prediction.shared_resources import SharedResources

"""
Tests for class ftio/prediction/shared_resources.py
"""


def test_shared_resources_init():
    sr = SharedResources()

    assert sr.queue is not None
    assert sr.data is not None
    assert sr.aggregated_bytes.value == 0.0
    assert sr.hits.value == 0.0
    assert sr.start_time.value == 0.0
    assert sr.count.value == 0

    sr.shutdown()


def test_shared_resources_queue():
    sr = SharedResources()

    sr.queue.put({"test": "data"})
    assert not sr.queue.empty()

    item = sr.queue.get()
    assert item == {"test": "data"}
    assert sr.queue.empty()

    sr.shutdown()


def test_shared_resources_update():
    sr = SharedResources()

    sr.aggregated_bytes.value = 1337.0
    sr.hits.value = 5.0
    sr.count.value = 10

    assert sr.aggregated_bytes.value == 1337.0
    assert sr.hits.value == 5.0
    assert sr.count.value == 10

    sr.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
