import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.common.debug import find_any, minimal
from tests.array_api.xputils import xp, xps

pytestmark = [pytest.mark.mockable_xp]


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
