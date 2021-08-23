import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument, Unsatisfiable

from hypothesis.extra.array_api import DTYPE_NAMES, NUMERIC_NAMES

from tests.common.debug import find_any, minimal
from tests.common.utils import fails_with, flaky
from tests.array_api.xputils import COMPLIANT_XP, xp, xps

pytestmark = [pytest.mark.mockable_xp]


def assert_array_namespace(array):
    """Check array has __array_namespace__() and it returns the correct module.

    This check is skipped if a mock array module is being used.
    """
    if COMPLIANT_XP:
        assert array.__array_namespace__() is xp


@given(xps.scalar_dtypes(), st.data())
def test_draw_arrays_from_dtype(dtype, data):
    """Draw arrays from dtypes."""
    array = data.draw(xps.arrays(dtype, ()))
    assert array.dtype == dtype
    assert_array_namespace(array)


@given(st.sampled_from(DTYPE_NAMES), st.data())
def test_draw_arrays_from_scalar_names(name, data):
    """Draw arrays from dtype names."""
    array = data.draw(xps.arrays(name, ()))
    assert array.dtype == getattr(xp, name)
    assert_array_namespace(array)


@given(xps.array_shapes(), st.data())
def test_can_draw_arrays_from_shapes(shape, data):
    """Draw arrays from shapes."""
    array = data.draw(xps.arrays(xp.int8, shape))
    assert array.ndim == len(shape)
    assert array.shape == shape
    assert_array_namespace(array)


@given(st.integers(0, 10), st.data())
def test_draw_arrays_from_int_shapes(size, data):
    """Draw arrays from integers as shapes."""
    array = data.draw(xps.arrays(xp.int8, size))
    assert array.shape == (size,)
    assert_array_namespace(array)


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
    array = data.draw(xps.arrays(strat, ()))
    assert_array_namespace(array)


@given(st.lists(st.sampled_from(DTYPE_NAMES), min_size=1, unique=True), st.data())
def test_draw_arrays_from_dtype_name_strategies(names, data):
    """Draw arrays from dtype name strategies."""
    names_strategy = st.sampled_from(names)
    array = data.draw(xps.arrays(names_strategy, ()))
    assert_array_namespace(array)


@given(xps.arrays(xp.int8, xps.array_shapes()))
def test_generate_arrays_from_shapes_strategy(array):
    """Generate arrays from shapes strategy."""
    assert_array_namespace(array)


@given(xps.arrays(xp.int8, st.integers(0, 100)))
def test_generate_arrays_from_integers_strategy_as_shape(array):
    """Generate arrays from integers strategy as shapes strategy."""
    assert_array_namespace(array)


@given(xps.arrays(xp.int8, ()))
def test_generate_arrays_from_zero_dimensions(array):
    """Generate arrays from empty shape."""
    assert array.shape == ()
    assert_array_namespace(array)


@given(xps.arrays(xp.int8, (1, 0, 1)))
def test_handle_zero_dimensions(array):
    """Generate arrays from empty shape."""
    assert array.shape == (1, 0, 1)
    assert_array_namespace(array)


@given(xps.arrays(xp.uint32, (5, 5)))
def test_generate_arrays_from_unsigned_ints(array):
    """Generate arrays from unsigned integer dtype."""
    assert xp.all(array >= 0)
    assert_array_namespace(array)


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


def test_minimize_large_uint_arrays():
    """Strategy with uint dtype and largely sized shape minimizes to a good
    example."""
    smallest = minimal(
        xps.arrays(xp.uint8, 100),
        lambda x: xp.any(x) and not xp.all(x),
        timeout_after=60,
    )
    assert xp.all(xp.logical_or(smallest == 0, smallest == 1))
    if hasattr(xp, "nonzero"):
        idx = xp.nonzero(smallest)[0]
        assert idx.size in (1, smallest.size - 1)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
@flaky(max_runs=50, min_passes=1)
def test_can_minimize_float_arrays():
    """Strategy with float dtype minimizes to a good example.

    We filter runtime warnings and expect flaky array generation for
    specifically NumPy - this behaviour may not be required when testing
    with other array libraries.
    """
    smallest = minimal(xps.arrays(xp.float32, 50), lambda x: xp.sum(x) >= 1.0)
    assert xp.sum(smallest) in (1, 50)


def count_unique(array):
    """Returns the number of unique elements.
    NaN values are treated as unique to eachother.

    The Array API doesn't specify how ``unique()`` should behave for Nan values,
    so this method provides consistent behaviour.
    """
    n_unique = 0

    nan_index = xp.isnan(array)
    for isnan, count in zip(*xp.unique(nan_index, return_counts=True)):
        if isnan:
            n_unique += count
            break

    # TODO: The Array API makes boolean indexing optinal, so in the future this
    # will need to be reworked if we want to test libraries other than NumPy.
    # If not possible, errors should be caught and the test skipped.
    filtered_array = array[~nan_index]
    unique_array = xp.unique(filtered_array)
    n_unique += unique_array.size

    return n_unique


@given(xps.arrays(xp.int8, st.integers(0, 20), unique=True))
def test_array_values_are_unique(array):
    if hasattr(xp, "unique"):
        assert count_unique(array) == array.size


def test_cannot_generate_unique_array_of_too_many_elements():
    strat = xps.arrays(xp.int8, 10, elements=st.integers(0, 5), unique=True)
    with pytest.raises(Unsatisfiable):
        strat.example()


def test_cannot_fill_with_non_castable_value():
    strat = xps.arrays(xp.int8, 10, fill=st.just("not a castable value"))
    with pytest.raises(InvalidArgument):
        strat.example()


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=st.integers(0, 20),
        elements=st.just(0.0),
        fill=st.just(xp.nan),
        unique=True,
    )
)
def test_array_values_are_unique_high_collision(array):
    hits = array == 0.0
    nzeros = 0
    for i in range(array.size):
        if hits[i]:
            nzeros += 1
    assert nzeros <= 1


@given(xps.arrays(xp.int8, (4,), elements=st.integers(0, 3), unique=True))
def test_generates_all_values_for_unique_array(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_may_fill_with_nan_when_unique_is_set():
    find_any(
        xps.arrays(
            dtype=xp.float32,
            shape=10,
            elements=st.floats(allow_nan=False),
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
        elements=st.floats(allow_nan=False),
        unique=True,
        fill=st.just(0.0),
    )
)
def test_may_not_fill_with_non_nan_when_unique_is_set(_):
    pass


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
    strat = xps.arrays(dtype=xp.int8, shape=1, **kwargs)
    data.draw(strat)


@pytest.mark.parametrize("fill", [False, True])
@pytest.mark.parametrize(
    "dtype, strat",
    [
        (xp.float32, st.floats(min_value=10 ** 40, allow_infinity=False)),
        (xp.float64, st.floats(min_value=10 ** 40, allow_infinity=False)),
    ],
)
@fails_with(InvalidArgument)
@given(st.data())
def test_may_not_use_unrepresentable_elements(fill, dtype, strat, data):
    if fill:
        kw = {"elements": st.nothing(), "fill": strat}
    else:
        kw = {"elements": strat}
    strat = xps.arrays(dtype=dtype, shape=1, **kw)
    data.draw(strat)


@given(
    xps.arrays(dtype=xp.float32, shape=10, elements={"min_value": 0, "max_value": 1})
)
def test_floats_can_be_constrained_at_low_width(array):
    assert xp.all(array >= 0)
    assert xp.all(array <= 1)


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
def test_floats_can_be_constrained_at_low_width_excluding_endpoints(array):
    assert xp.all(array > 0)
    assert xp.all(array < 1)


@given(
    xps.arrays(
        dtype=xp.float32,
        elements=st.just(xp.nan),
        shape=xps.array_shapes(),
    )
)
def test_count_unique(array):
    assert count_unique(array) == array.size


@given(
    xps.arrays(
        dtype=xp.float32,
        elements=st.floats(allow_nan=False, width=32),
        shape=10,
        unique=True,
        fill=st.just(xp.nan),
    )
)
def test_is_still_unique_with_nan_fill(array):
    if hasattr(xp, "unique"):
        assert count_unique(array) == array.size


@given(
    xps.arrays(
        dtype=xp.float32,
        shape=10,
        unique=True,
        elements=st.integers(1, 9),
        fill=st.just(xp.nan),
    )
)
def test_unique_array_with_fill_can_use_all_elements(array):
    if hasattr(xp, "unique"):
        assume(count_unique(array) == array.size)


@given(xps.arrays(dtype=xp.uint8, shape=25, unique=True, fill=st.nothing()))
def test_unique_array_without_fill(array):
    # This test covers the collision-related branches for fully dense unique arrayays.
    # Choosing 25 of 256 possible elements means we're almost certain to see colisions
    # thanks to the 'birthday paradox', but finding unique elemennts is still easy.
    if hasattr(xp, "unique"):
        assume(count_unique(array) == array.size)


@given(xps.arrays(xp.bool, (), fill=st.nothing()))
def test_can_generate_0d_arrays_with_no_fill(array):
    assert array.dtype == xp.bool
    assert array.ndim == 0
    assert array.shape == ()


@st.composite
def distinct_integers(draw):
    used = draw(st.shared(st.builds(set), key="distinct_integers.used"))
    i = draw(st.integers(0, 2 ** 64 - 1).filter(lambda x: x not in used))
    used.add(i)
    return i


@given(xps.arrays(xp.uint64, 10, elements=distinct_integers()))
def test_does_not_reuse_distinct_integers(array):
    # xp.unique() is optional for Array API libraries
    if hasattr(xp, "unique"):
        unique_values = xp.unique(array)
        assert unique_values.size == array.size


def test_may_reuse_distinct_integers_if_asked():
    if hasattr(xp, "unique"):

        def nunique(array) -> int:
            unique_values = xp.unique(array)
            return unique_values.size

        find_any(
            xps.arrays(
                xp.uint64, 10, elements=distinct_integers(), fill=distinct_integers()
            ),
            lambda x: nunique(x) < len(x),
        )
    else:
        pytest.skip()


def test_minimizes_to_fill():
    smallest = minimal(xps.arrays(xp.float32, 10, fill=st.just(3.0)))
    assert xp.all(smallest == 3.0)


@given(
    xps.arrays(
        dtype=xp.float32,
        elements=st.floats(width=32).filter(bool),
        shape=(3, 3, 3),
        fill=st.just(1.0),
    )
)
def test_fills_everything(array):
    assert xp.all(array)


@pytest.mark.parametrize("dtype", [xp.float32, xp.float64])
@pytest.mark.parametrize("low", [-2.0, -1.0, 0.0, 1.0])
@given(st.data())
def test_bad_float_exclude_min_in_array(dtype, low, data):
    strat = xps.arrays(
        dtype=dtype,
        shape=(),
        elements={
            "min_value": low,
            "max_value": low + 1,
            "exclude_min": True,
        },
    )
    array = data.draw(strat, label="array")
    assert array > low
