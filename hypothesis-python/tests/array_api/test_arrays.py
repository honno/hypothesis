# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Copyright the Hypothesis Authors.
# Individual contributors are listed in AUTHORS.rst and the git log.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from hypothesis import assume, given, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.extra.array_api import DTYPE_NAMES, NUMERIC_NAMES
from hypothesis.internal.floats import width_smallest_normals

from tests.array_api.common import COMPLIANT_XP, WIDTHS_FTZ, xp, xps
from tests.common.debug import find_any, minimal
from tests.common.utils import fails_with, flaky

needs_xp_unique_values = pytest.mark.skipif(
    not hasattr(xp, "unique_values"), reason="optional API"
)


# xp.unique_value() should return distinct NaNs - if not, tests that (rightly)
# assume such behaviour will likely fail. This mark namely addresses mocking the
# array module with NumPy 1.21, which treats NaNs as not distinct.
# See https://mail.python.org/pipermail/numpy-discussion/2021-August/081995.html
two_nans = xp.asarray([float("nan"), float("nan")])
assumes_distinct_nans = pytest.mark.xfail(
    hasattr(xp, "unique_values") and xp.unique_values(two_nans).size != 2,
    reason="NaNs not distinct",
)


def assert_array_namespace(x):
    """Check array has __array_namespace__() and it returns the correct module.

    This check is skipped if a mock array module is being used.
    """
    if COMPLIANT_XP:
        assert x.__array_namespace__() is xp


@given(xps.scalar_dtypes(), st.data())
def test_draw_arrays_from_dtype(dtype, data):
    """Draw arrays from dtypes."""
    x = data.draw(xps.arrays(dtype, ()))
    assert x.dtype == dtype
    assert_array_namespace(x)


@given(st.sampled_from(DTYPE_NAMES), st.data())
def test_draw_arrays_from_scalar_names(name, data):
    """Draw arrays from dtype names."""
    x = data.draw(xps.arrays(name, ()))
    assert x.dtype == getattr(xp, name)
    assert_array_namespace(x)


@given(xps.array_shapes(), st.data())
def test_draw_arrays_from_shapes(shape, data):
    """Draw arrays from shapes."""
    x = data.draw(xps.arrays(xp.int8, shape))
    assert x.ndim == len(shape)
    assert x.shape == shape
    assert_array_namespace(x)


@given(st.integers(0, 10), st.data())
def test_draw_arrays_from_int_shapes(size, data):
    """Draw arrays from integers as shapes."""
    x = data.draw(xps.arrays(xp.int8, size))
    assert x.shape == (size,)
    assert_array_namespace(x)


@pytest.mark.parametrize(
    "strat",
    [
        xps.scalar_dtypes(),
        xps.boolean_dtypes(),
        xps.integer_dtypes(),
        xps.unsigned_integer_dtypes(),
        xps.floating_dtypes(),
    ],
)
@given(st.data())
def test_draw_arrays_from_dtype_strategies(strat, data):
    """Draw arrays from dtype strategies."""
    x = data.draw(xps.arrays(strat, ()))
    assert_array_namespace(x)


@given(st.lists(st.sampled_from(DTYPE_NAMES), min_size=1, unique=True), st.data())
def test_draw_arrays_from_dtype_name_strategies(names, data):
    """Draw arrays from dtype name strategies."""
    names_strategy = st.sampled_from(names)
    x = data.draw(xps.arrays(names_strategy, ()))
    assert_array_namespace(x)


@given(xps.arrays(xp.int8, xps.array_shapes()))
def test_generate_arrays_from_shapes_strategy(x):
    """Generate arrays from shapes strategy."""
    assert_array_namespace(x)


@given(xps.arrays(xp.int8, st.integers(0, 100)))
def test_generate_arrays_from_integers_strategy_as_shape(x):
    """Generate arrays from integers strategy as shapes strategy."""
    assert_array_namespace(x)


@given(xps.arrays(xp.int8, ()))
def test_generate_arrays_from_zero_dimensions(x):
    """Generate arrays from empty shape."""
    assert x.shape == ()
    assert_array_namespace(x)


@given(xps.arrays(xp.int8, (1, 0, 1)))
def test_handle_zero_dimensions(x):
    """Generate arrays from empty shape."""
    assert x.shape == (1, 0, 1)
    assert_array_namespace(x)


@given(xps.arrays(xp.uint32, (5, 5)))
def test_generate_arrays_from_unsigned_ints(x):
    """Generate arrays from unsigned integer dtype."""
    assert xp.all(x >= 0)
    assert_array_namespace(x)


@given(
    xps.arrays(
        dtype=xp.uint8,
        shape=(5, 5),
        elements=xps.from_dtype(xp.uint8).map(lambda e: xp.asarray(e, dtype=xp.uint8)),
    )
)
def test_generate_arrays_from_0d_arrays(x):
    """Generate arrays from 0d array elements."""
    assert x.shape == (5, 5)
    assert_array_namespace(x)


def test_minimize_arrays_with_default_dtype_shape_strategies():
    """Strategy with default scalar_dtypes and array_shapes strategies minimize
    to a boolean 1-dimensional array of size 1."""
    smallest = minimal(xps.arrays(xps.scalar_dtypes(), xps.array_shapes()))
    assert smallest.shape == (1,)
    assert smallest.dtype == xp.bool
    assert not xp.any(smallest)


def test_minimize_arrays_with_0d_shape_strategy():
    """Strategy with shape strategy that can generate empty tuples minimizes to
    0d arrays."""
    smallest = minimal(xps.arrays(xp.int8, xps.array_shapes(min_dims=0)))
    assert smallest.shape == ()


@pytest.mark.parametrize("dtype", NUMERIC_NAMES)
def test_minimizes_numeric_arrays(dtype):
    """Strategies with numeric dtypes minimize to zero-filled arrays."""
    smallest = minimal(xps.arrays(dtype, (2, 2)))
    assert xp.all(smallest == 0)


@pytest.mark.skipif(not hasattr(xp, "nonzero"), reason="optional API")
def test_minimize_large_uint_arrays():
    """Strategy with uint dtype and largely sized shape minimizes to a good
    example."""
    smallest = minimal(
        xps.arrays(xp.uint8, 100),
        lambda x: xp.any(x) and not xp.all(x),
        timeout_after=60,
    )
    assert xp.all(xp.logical_or(smallest == 0, smallest == 1))
    idx = xp.nonzero(smallest)[0]
    assert idx.size in (1, smallest.size - 1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@flaky(max_runs=50, min_passes=1)
def test_minimize_float_arrays():
    """Strategy with float dtype minimizes to a good example.

    We filter runtime warnings and expect flaky array generation for
    specifically NumPy - this behaviour may not be required when testing
    with other array libraries.
    """
    smallest = minimal(xps.arrays(xp.float32, 50), lambda x: xp.sum(x) >= 1.0)
    assert xp.sum(smallest) in (1, 50)


def test_minimizes_to_fill():
    """Strategy with single fill value minimizes to arrays only containing said
    fill value."""
    smallest = minimal(xps.arrays(xp.float32, 10, fill=st.just(3.0)))
    assert xp.all(smallest == 3.0)


@needs_xp_unique_values
@given(xps.arrays(xp.int8, st.integers(0, 20), unique=True))
def test_generate_unique_arrays(x):
    """Generates unique arrays."""
    assert xp.unique_values(x).size == x.size


@fails_with(InvalidArgument)
@given(xps.arrays(xp.int8, 10, elements=st.integers(0, 5), unique=True))
def test_cannot_draw_unique_arrays_with_too_small_elements(_):
    """Unique strategy with elements strategy range smaller than its size raises
    helpful error."""


@fails_with(InvalidArgument)
@given(xps.arrays(xp.int8, 10, fill=st.just("not a castable value")))
def test_cannot_fill_arrays_with_non_castable_value():
    """Strategy with fill not castable to dtype raises helpful error."""


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=st.integers(0, 20),
        elements=st.just(0.0),
        fill=st.just(xp.nan),
        unique=True,
    )
)
def test_generate_unique_arrays_with_high_collision_elements(x):
    """Generates unique arrays with just elements of 0.0 and NaN fill."""
    zero_mask = x == 0.0
    assert xp.sum(xp.astype(zero_mask, xp.uint8)) <= 1


@needs_xp_unique_values
@given(xps.arrays(xp.int8, (4,), elements=st.integers(0, 3), unique=True))
def test_generate_unique_arrays_using_all_elements(x):
    """Unique strategy with elements strategy range equal to its size will only
    generate arrays with one of each possible element."""
    assert xp.unique_values(x).size == x.size


def test_may_fill_unique_arrays_with_nan():
    """Unique strategy with NaN fill can generate arrays holding NaNs."""
    find_any(
        xps.arrays(
            dtype=xp.float32,
            shape=10,
            elements={"allow_nan": False},
            unique=True,
            fill=st.just(xp.nan),
        ),
        lambda x: xp.any(xp.isnan(x)),
    )


@fails_with(InvalidArgument)
@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        elements={"allow_nan": False},
        unique=True,
        fill=st.just(0.0),
    )
)
def test_may_not_fill_unique_array_with_non_nan(_):
    """Unique strategy with just fill elements of 0.0 raises helpful error."""


@pytest.mark.parametrize(
    "kwargs",
    [
        {"elements": st.just(300)},
        {"elements": st.nothing(), "fill": st.just(300)},
    ],
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_overflowing_integers(kwargs, data):
    """Strategy with elements strategy range outside the dtype's bounds raises
    helpful error."""
    strat = xps.arrays(dtype=xp.int8, shape=1, **kwargs)
    data.draw(strat)


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize(
    "dtype, strat",
    [
        (xp.float32, st.floats(min_value=10**40, allow_infinity=False)),
        (xp.float64, st.floats(min_value=10**40, allow_infinity=False)),
    ],
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_unrepresentable_elements(fill, dtype, strat, data):
    """Strategy with elements not representable by the dtype raises helpful error."""
    if fill:
        kw = {"elements": st.nothing(), "fill": strat}
    else:
        kw = {"elements": strat}
    strat = xps.arrays(dtype=dtype, shape=1, **kw)
    data.draw(strat)


@given(
    xps.arrays(dtype=xp.float32, shape=10, elements={"min_value": 0, "max_value": 1})
)
def test_floats_can_be_constrained(x):
    """Strategy with float dtype and specified elements strategy range
    (inclusive) generates arrays with elements inside said range."""
    assert xp.all(x >= 0)
    assert xp.all(x <= 1)


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        elements={
            "min_value": 0,
            "max_value": 1,
            "exclude_min": True,
            "exclude_max": True,
        },
    )
)
def test_floats_can_be_constrained_excluding_endpoints(x):
    """Strategy with float dtype and specified elements strategy range
    (exclusive) generates arrays with elements inside said range."""
    assert xp.all(x > 0)
    assert xp.all(x < 1)


@needs_xp_unique_values
@assumes_distinct_nans
@given(
    xps.arrays(
        dtype=xp.float32,
        elements={"allow_nan": False},
        shape=10,
        unique=True,
        fill=st.just(xp.nan),
    )
)
def test_is_still_unique_with_nan_fill(x):
    """Unique strategy with NaN fill generates unique arrays."""
    assert xp.unique_values(x).size == x.size


@needs_xp_unique_values
@assumes_distinct_nans
@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        unique=True,
        elements=st.integers(1, 9),
        fill=st.just(xp.nan),
    )
)
def test_unique_array_with_fill_can_use_all_elements(x):
    """Unique strategy with elements range equivalent to its size and NaN fill
    can generate arrays with all possible values."""
    assume(xp.unique_values(x).size == x.size)


@needs_xp_unique_values
@given(xps.arrays(dtype=xp.uint8, shape=25, unique=True, fill=st.nothing()))
def test_generate_unique_arrays_without_fill(x):
    """Generate arrays from unique strategy with no fill.

    Covers the collision-related branches for fully dense unique arrays.
    Choosing 25 of 256 possible values means we're almost certain to see
    colisions thanks to the birthday paradox, but finding unique values should
    still be easy.
    """
    assume(xp.unique_values(x).size == x.size)


@needs_xp_unique_values
@given(xps.arrays(dtype=xp.int8, shape=255, unique=True))
def test_efficiently_generate_unique_arrays_using_all_elements(x):
    """Unique strategy with elements strategy range equivalent to its size
    generates arrays with all possible values. Generation is not too slow.

    Avoids the birthday paradox with UniqueSampledListStrategy.
    """
    assert xp.unique_values(x).size == x.size


@needs_xp_unique_values
@given(st.data(), st.integers(-100, 100), st.integers(1, 100))
def test_array_element_rewriting(data, start, size):
    """Unique strategy generates arrays with expected elements."""
    x = data.draw(
        xps.arrays(
            dtype=xp.int64,
            shape=size,
            elements=st.integers(start, start + size - 1),
            unique=True,
        )
    )
    x_set_expect = xp.linspace(start, start + size - 1, size, dtype=xp.int64)
    x_set = xp.sort(xp.unique_values(x))
    assert xp.all(x_set == x_set_expect)


@given(xps.arrays(xp.bool, (), fill=st.nothing()))
def test_generate_0d_arrays_with_no_fill(x):
    """Generate arrays with zero-dimensions and no fill."""
    assert x.dtype == xp.bool
    assert x.shape == ()


@pytest.mark.parametrize("dtype", [xp.float32, xp.float64])
@pytest.mark.parametrize("low", [-2.0, -1.0, 0.0, 1.0])
@given(st.data())
def test_excluded_min_in_float_arrays(dtype, low, data):
    """Strategy with elements strategy excluding min does not generate arrays
    with elements less or equal to said min."""
    strat = xps.arrays(
        dtype=dtype,
        shape=(),
        elements={
            "min_value": low,
            "max_value": low + 1,
            "exclude_min": True,
        },
    )
    x = data.draw(strat, label="array")
    assert xp.all(x > low)


@st.composite
def distinct_integers(draw):
    used = draw(st.shared(st.builds(set), key="distinct_integers.used"))
    i = draw(st.integers(0, 2**64 - 1).filter(lambda x: x not in used))
    used.add(i)
    return i


@needs_xp_unique_values
@given(xps.arrays(xp.uint64, 10, elements=distinct_integers()))
def test_does_not_reuse_distinct_integers(x):
    """Strategy with distinct integer elements strategy generates arrays with
    distinct values."""
    assert xp.unique_values(x).size == x.size


@needs_xp_unique_values
def test_may_reuse_distinct_integers_if_asked():
    """Strategy with shared elements and fill strategies of distinct integers
    may generate arrays with non-distinct values."""
    find_any(
        xps.arrays(
            xp.uint64, 10, elements=distinct_integers(), fill=distinct_integers()
        ),
        lambda x: xp.unique_values(x).size < x.size,
    )


@pytest.mark.skipif(
    not WIDTHS_FTZ[32], reason="Subnormals are valid for non-FTZ builds"
)
def test_cannot_draw_subnormals_for_ftz_float32():
    """For FTZ builds of array modules, strategy with subnormal elements
    strategy raises helpful error."""
    strat = xps.arrays(
        xp.float32,
        10,
        elements={
            "min_value": 0.0,
            "max_value": width_smallest_normals[32],
            "exclude_min": True,
            "exclude_max": True,
            "allow_subnormal": True,
        },
    )
    with pytest.raises(InvalidArgument, match="Generated subnormal float"):
        strat.example()
