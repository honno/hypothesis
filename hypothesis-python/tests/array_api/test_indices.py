import math

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from tests.common.debug import find_any
from tests.array_api.xputils import xp, xps

pytestmark = [pytest.mark.mockable_xp]


@pytest.mark.parametrize(
    "condition",
    [
        lambda ix: Ellipsis in ix,
        lambda ix: Ellipsis not in ix,
    ],
)
def test_indices_options(condition):
    indexers = xps.array_shapes(min_dims=1, max_dims=32).flatmap(
        lambda shape: xps.indices(shape, allow_none=True)
    )
    find_any(indexers, condition)


def test_indices_can_generate_empty_tuple():
    find_any(xps.indices(shape=(0, 0), allow_ellipsis=True), lambda ix: ix == ())


def test_indices_can_generate_non_tuples():
    find_any(
        xps.indices(shape=(0, 0), allow_ellipsis=True),
        lambda ix: not isinstance(ix, tuple),
    )


def test_indices_can_generate_long_ellipsis():
    # Runs of slice(None) - such as [0,:,:,:,0] - can be replaced by e.g. [0,...,0]
    find_any(
        xps.indices(shape=(1, 0, 0, 0, 1), allow_ellipsis=True),
        lambda ix: len(ix) == 3 and ix[1] == Ellipsis,
    )


@given(
    xps.indices(shape=(0, 0, 0, 0, 0)).filter(
        lambda idx: isinstance(idx, tuple) and Ellipsis in idx
    )
)
def test_indices_replaces_whole_axis_slices_with_ellipsis(idx):
    # If ... is in the slice, it replaces all ,:, entries for this shape.
    assert slice(None) not in idx


@given(
    shape=xps.array_shapes(min_dims=1, max_side=4)
    | xps.array_shapes(min_dims=1, min_side=0, max_side=10),
    min_dims=st.integers(1, 5),
    allow_ellipsis=st.booleans(),
    allow_none=st.booleans(),
    data=st.data(),
)
def test_indices_generate_valid_indexers(
    shape, min_dims, allow_ellipsis, allow_none, data
):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    indexer = data.draw(
        xps.indices(
            shape,
            min_dims=min_dims,
            max_dims=max_dims,
            allow_ellipsis=allow_ellipsis,
            allow_none=allow_none,
        ),
        label="indexer",
    )
    # Check that disallowed things are indeed absent
    if not allow_none:
        if isinstance(indexer, tuple):
            assert 0 <= len(indexer) <= len(shape) + int(allow_ellipsis)
        else:
            assert 1 <= len(shape) + int(allow_ellipsis)
        assert None not in shape
    if not allow_ellipsis:
        assert Ellipsis not in shape

    if 0 in shape:
        # If there's a zero in the shape, the array will have no elements.
        array = xp.zeros(shape)
        assert array.size == 0
    elif math.prod(shape) <= 10 ** 5:
        # If it's small enough to instantiate, do so with distinct elements.
        array = xp.reshape(xp.arange(math.prod(shape)), shape)
    else:
        # We can't cheat on this one, so just try another.
        assume(False)
