# This file is part of Hypothesis, which may be found at
# https://github.com/HypothesisWorks/hypothesis/
#
# Most of this work is copyright (C) 2013-2021 David R. MacIver
# (david@drmaciver.com), but it contains contributions by others. See
# CONTRIBUTING.rst for a full list of people who may hold copyright, and
# consult the git log if you need to determine who owns an individual
# contribution.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.
#
# END HEADER

from functools import reduce
from itertools import zip_longest

import pytest

from hypothesis import HealthCheck, assume, given, note, settings, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.extra.array_api import Shape

from tests.array_api.xputils import xp, xps
from tests.common.debug import find_any, minimal

pytestmark = [pytest.mark.mockable_xp]

ANY_SHAPE = xps.array_shapes(min_dims=0, max_dims=32, min_side=0, max_side=32)
ANY_NONZERO_SHAPE = xps.array_shapes(min_dims=0, max_dims=32, min_side=1, max_side=32)


@given(xps.array_shapes())
def test_can_generate_array_shapes(shape):
    assert isinstance(shape, tuple)
    assert all(isinstance(i, int) for i in shape)


@settings(deadline=None, max_examples=10)
@given(st.integers(0, 10), st.integers(0, 9), st.integers(0), st.integers(0))
def test_minimise_array_shapes(min_dims, dim_range, min_side, side_range):
    smallest = minimal(
        xps.array_shapes(
            min_dims=min_dims,
            max_dims=min_dims + dim_range,
            min_side=min_side,
            max_side=min_side + side_range,
        )
    )
    assert len(smallest) == min_dims and all(k == min_side for k in smallest)


@pytest.mark.parametrize(
    "kwargs", [{"min_side": 100}, {"min_dims": 15}, {"min_dims": 32}]
)
def test_interesting_array_shapes_argument(kwargs):
    xps.array_shapes(**kwargs).example()


@given(xps.broadcastable_shapes((), min_side=0, max_side=0, min_dims=0, max_dims=0))
def test_broadcastable_empty_shape(shape):
    assert shape == ()


@given(shape=ANY_SHAPE, data=st.data())
@settings(deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_broadcastable_shape_bounds_are_satisfied(shape, data):
    min_dims = data.draw(st.integers(0, 32), label="min_dims")
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side = data.draw(st.integers(0, 3), label="min_side")
    max_side = data.draw(st.none() | st.integers(min_side, 6), label="max_side")
    try:
        bshape = data.draw(
            xps.broadcastable_shapes(
                shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims,
            ),
            label="bshape",
        )
    except InvalidArgument:
        assume(False)
        assert False, "unreachable"

    if max_dims is None:
        max_dims = max(len(shape), min_dims) + 2

    if max_side is None:
        max_side = max(tuple(shape[::-1][:max_dims]) + (min_side,)) + 2

    assert isinstance(bshape, tuple) and all(isinstance(s, int) for s in bshape)
    assert min_dims <= len(bshape) <= max_dims
    assert all(min_side <= s <= max_side for s in bshape)


def _draw_valid_bounds(data, shape, max_dims, permit_none=True):
    if max_dims == 0 or not shape:
        return 0, None

    smallest_side = min(shape[::-1][:max_dims])
    if smallest_side > 1:
        min_strat = st.sampled_from([1, smallest_side])
    else:
        min_strat = st.just(smallest_side)
    min_side = data.draw(min_strat, label="min_side")

    largest_side = max(max(shape[::-1][:max_dims]), min_side)
    if permit_none:
        max_strat = st.one_of(st.none(), st.integers(largest_side, largest_side + 2))
    else:
        max_strat = st.integers(largest_side, largest_side + 2)
    max_side = data.draw(max_strat, label="max_side")

    return min_side, max_side


def _broadcast_two_shapes(shape1: Shape, shape2: Shape) -> Shape:
    result = []
    for a, b in zip_longest(shape1[::-1], shape2[::-1], fillvalue=1):
        if a != b and (a != 1) and (b != 1):
            raise ValueError(
                f"shapes {shape1} and {shape2} are not broadcast-compatible"
            )
        result.append(a if a != 1 else b)
    return tuple(result[::-1])


def _broadcast_shapes(*shapes):
    """Returns the shape resulting from broadcasting the input shapes together.

    Raises `ValueError` if the shapes are not broadcast-compatible"""
    assert shapes, "Must pass >=1 shapes to broadcast"
    return reduce(_broadcast_two_shapes, shapes, ())


@settings(deadline=None, max_examples=500)
@given(
    shapes=st.lists(
        xps.array_shapes(min_dims=0, min_side=0, max_dims=4, max_side=4), min_size=1
    )
)
def test_broadcastable_shape_util(shapes):
    # Ensures that _broadcast_shapes() raises when fed incompatible shapes,
    # and ensures that it produces the true broadcasted shape
    if len(shapes) == 1:
        assert _broadcast_shapes(*shapes) == shapes[0]
        return

    arrays = [xp.zeros(s, dtype=xp.uint8) for s in shapes]

    try:
        broadcast_out = xp.broadcast_arrays(*arrays)
    except ValueError:
        with pytest.raises(ValueError):
            _broadcast_shapes(*shapes)
        return
    broadcasted_shape = _broadcast_shapes(*shapes)

    assert broadcast_out[0].shape == broadcasted_shape


@given(shape=ANY_NONZERO_SHAPE, data=st.data())
@settings(deadline=None, max_examples=200)
def test_broadcastable_shape_has_good_default_values(shape, data):
    # Ensures that default parameters can always produce broadcast-compatible shapes
    broadcastable_shape = data.draw(
        xps.broadcastable_shapes(shape), label="broadcastable_shapes"
    )
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), shape=ANY_SHAPE, data=st.data())
@settings(deadline=None)
def test_broadcastable_shape_can_broadcast(min_dims, shape, data):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side, max_side = _draw_valid_bounds(data, shape, max_dims)
    broadcastable_shape = data.draw(
        xps.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        ),
        label="broadcastable_shapes",
    )
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), shape=ANY_SHAPE, data=st.data())
@settings(deadline=None, max_examples=10)
def test_minimize_broadcastable_shape(min_dims, shape, data):
    # Ensure aligned dimensions of broadcastable shape minimizes to `(1,) * min_dims`
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side, max_side = _draw_valid_bounds(data, shape, max_dims, permit_none=False)
    smallest = minimal(
        xps.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    note(f"(smallest): {smallest}")
    n_leading = max(len(smallest) - len(shape), 0)
    n_aligned = max(len(smallest) - n_leading, 0)
    expected = [min_side] * n_leading + [
        1 if min_side <= 1 <= max_side else i for i in shape[len(shape) - n_aligned :]
    ]
    assert tuple(expected) == smallest


@given(max_dims=st.integers(4, 6), data=st.data())
@settings(deadline=None)
def test_broadcastable_shape_adjusts_max_dim_with_explicit_bounds(max_dims, data):
    # Ensures that broadcastable_shapes() limits itself to satisfiable dimensions
    # Broadcastable values can only be drawn for dims 0-3 for these shapes
    shape = data.draw(st.sampled_from([(5, 3, 2, 1), (0, 3, 2, 1)]), label="shape")
    broadcastable_shape = data.draw(
        xps.broadcastable_shapes(
            shape, min_side=2, max_side=3, min_dims=3, max_dims=max_dims
        ),
        label="broadcastable_shapes",
    )
    assert len(broadcastable_shape) == 3
    # error if drawn shape for b is not broadcast-compatible
    _broadcast_shapes(shape, broadcastable_shape)


@given(min_dims=st.integers(0, 32), min_side=st.integers(2, 3), data=st.data())
@settings(deadline=None, max_examples=10)
def test_broadcastable_shape_shrinking_with_singleton_out_of_bounds(
    min_dims, min_side, data
):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    max_side = data.draw(st.none() | st.integers(min_side, 6), label="max_side")
    shape = data.draw(st.integers(1, 4).map(lambda n: n * (1,)), label="shape")
    smallest = minimal(
        xps.broadcastable_shapes(
            shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    assert smallest == (min_side,) * min_dims


@given(
    shape=xps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
    max_dims=st.integers(0, 6),
    data=st.data(),
)
@settings(deadline=None)
def test_broadcastable_shape_can_generate_arbitrary_ndims(shape, max_dims, data):
    # ensures that generates shapes can possess any length in [min_dims, max_dims]
    desired_ndim = data.draw(st.integers(0, max_dims), label="desired_ndim")
    min_dims = data.draw(
        st.one_of(st.none(), st.integers(0, desired_ndim)), label="min_dims"
    )
    # check default arg behavior too
    kwargs = {"min_dims": min_dims} if min_dims is not None else {}
    find_any(
        xps.broadcastable_shapes(shape, min_side=0, max_dims=max_dims, **kwargs),
        lambda x: len(x) == desired_ndim,
        settings(max_examples=10 ** 6),
    )


@given(num_shapes=st.integers(1, 4), base_shape=ANY_SHAPE, data=st.data())
@settings(deadline=None)
def test_mutually_broadcastable_shape_bounds_are_satisfied(
    num_shapes, base_shape, data
):
    min_dims = data.draw(st.integers(0, 32), label="min_dims")
    max_dims = data.draw(
        st.one_of(st.none(), st.integers(min_dims, 32)), label="max_dims"
    )
    min_side = data.draw(st.integers(0, 3), label="min_side")
    max_side = data.draw(
        st.one_of(st.none(), st.integers(min_side, 6)), label="max_side"
    )
    try:
        shapes, result = data.draw(
            xps.mutually_broadcastable_shapes(
                num_shapes=num_shapes,
                base_shape=base_shape,
                min_side=min_side,
                max_side=max_side,
                min_dims=min_dims,
                max_dims=max_dims,
            ),
            label="shapes, result",
        )
    except InvalidArgument:
        assume(False)
        assert False, "unreachable"

    if max_dims is None:
        max_dims = max(len(base_shape), min_dims) + 2

    if max_side is None:
        max_side = max(tuple(base_shape[::-1][:max_dims]) + (min_side,)) + 2

    assert isinstance(shapes, tuple)
    assert isinstance(result, tuple)
    assert all(isinstance(s, int) for s in result)

    for bshape in shapes:
        assert isinstance(bshape, tuple) and all(isinstance(s, int) for s in bshape)
        assert min_dims <= len(bshape) <= max_dims
        assert all(min_side <= s <= max_side for s in bshape)


@given(base_shape=ANY_SHAPE, num_shapes=st.integers(1, 10), data=st.data())
@settings(deadline=None, max_examples=200)
def test_mutually_broadcastableshapes_has_good_default_values(
    num_shapes, base_shape, data
):
    # ensures that default parameters can always produce broadcast-compatible shapes
    shapes, result = data.draw(
        xps.mutually_broadcastable_shapes(num_shapes=num_shapes, base_shape=base_shape),
        label="shapes, result",
    )
    assert len(shapes) == num_shapes
    # raises if shapes are not mutually-compatible
    assert result == _broadcast_shapes(base_shape, *shapes)


@given(
    num_shapes=st.integers(1, 10),
    min_dims=st.integers(0, 32),
    base_shape=ANY_SHAPE,
    data=st.data(),
)
@settings(deadline=None)
def test_mutually_broadcastable_shape_can_broadcast(
    num_shapes, min_dims, base_shape, data
):
    max_dims = data.draw(st.none() | st.integers(min_dims, 32), label="max_dims")
    min_side, max_side = _draw_valid_bounds(data, base_shape, max_dims)
    shapes, result = data.draw(
        xps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        ),
        label="shapes, result",
    )

    # error if drawn shapes are not mutually broadcast-compatible
    assert result == _broadcast_shapes(base_shape, *shapes)


@given(
    num_shapes=st.integers(1, 3),
    min_dims=st.integers(0, 5),
    base_shape=xps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
    data=st.data(),
)
@settings(deadline=None, max_examples=50)
def test_minimize_mutually_broadcastable_shape(num_shapes, min_dims, base_shape, data):
    # ensure aligned dimensions of broadcastable shape minimizes to (1,) * min_dims
    max_dims = data.draw(st.none() | st.integers(min_dims, 5), label="max_dims")
    min_side, max_side = _draw_valid_bounds(
        data, base_shape, max_dims, permit_none=False
    )

    if num_shapes > 1:
        # shrinking gets a little bit hairy when we have empty axes
        # and multiple num_shapes
        assume(min_side > 0)
    note(f"(min_side, max_side): {(min_side, max_side)}")
    smallest_shapes, result = minimal(
        xps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    note(f"(smallest_shapes, result): {(smallest_shapes, result)}")
    assert len(smallest_shapes) == num_shapes
    assert result == _broadcast_shapes(base_shape, *smallest_shapes)
    for smallest in smallest_shapes:
        n_leading = max(len(smallest) - len(base_shape), 0)
        n_aligned = max(len(smallest) - n_leading, 0)
        expected = [min_side] * n_leading + [
            1 if min_side <= 1 <= max_side else i
            for i in base_shape[len(base_shape) - n_aligned :]
        ]
        assert tuple(expected) == smallest


@given(
    max_side=st.sampled_from([3, None]),
    min_dims=st.integers(0, 4),
    num_shapes=st.integers(1, 3),
    data=st.data(),
)
@settings(deadline=None)
def test_mutually_broadcastable_shape_adjusts_max_dim_with_default_bounds(
    max_side, min_dims, num_shapes, data
):
    # Ensures that mutually_broadcastable_shapes limits itself to
    # satisfiable dimensions when a default max_dims is derived.
    base_shape = data.draw(
        st.sampled_from([(5, 3, 2, 1), (0, 3, 2, 1)]), label="base_shape"
    )

    try:
        shapes, result = data.draw(
            xps.mutually_broadcastable_shapes(
                num_shapes=num_shapes,
                base_shape=base_shape,
                min_side=2,
                max_side=max_side,
                min_dims=min_dims,
            ),
            label="shapes, result",
        )
    except InvalidArgument:
        # There is no satisfiable max_dims for us to tune
        assert min_dims == 4 and (max_side == 3 or base_shape[0] == 0)
        return

    if max_side == 3 or base_shape[0] == 0:
        assert all(len(s) <= 3 for s in shapes)
    elif min_dims == 4:
        assert all(4 <= len(s) for s in shapes)

    # error if drawn shape for b is not broadcast-compatible
    assert len(shapes) == num_shapes
    assert result == _broadcast_shapes(base_shape, *shapes)


@given(
    num_shapes=st.integers(1, 4),
    min_dims=st.integers(0, 4),
    min_side=st.integers(2, 3),
    data=st.data(),
)
@settings(deadline=None, max_examples=50)
def test_mutually_broadcastable_shapes_shrinking_with_singleton_out_of_bounds(
    num_shapes, min_dims, min_side, data
):
    # Ensures that shapes minimize to (min_side,) * min_dims
    # when singleton dimensions are disallowed.
    max_dims = data.draw(st.none() | st.integers(min_dims, 4), label="max_dims")
    max_side = data.draw(
        st.one_of(st.none(), st.integers(min_side, 6)), label="max_side"
    )
    ndims = data.draw(st.integers(1, 4), label="ndim")
    base_shape = (1,) * ndims
    smallest_shapes, result = minimal(
        xps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_side=min_side,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        )
    )
    note(f"(smallest_shapes, result): {(smallest_shapes, result)}")
    assert len(smallest_shapes) == num_shapes
    assert result == _broadcast_shapes(base_shape, *smallest_shapes)
    for smallest in smallest_shapes:
        assert smallest == (min_side,) * min_dims


@given(
    num_shapes=st.integers(1, 4),
    min_dims=st.integers(1, 32),
    max_side=st.integers(1, 6),
    data=st.data(),
)
def test_mutually_broadcastable_shapes_only_singleton_is_valid(
    num_shapes, min_dims, max_side, data
):
    # Ensures that, when all aligned base-shape dim sizes are
    # larger than max_side, only singletons can be drawn.
    max_dims = data.draw(st.integers(min_dims, 32), label="max_dims")
    base_shape = data.draw(
        xps.array_shapes(min_side=max_side + 1, min_dims=1), label="base_shape"
    )
    input_shapes, result = data.draw(
        xps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_side=1,
            max_side=max_side,
            min_dims=min_dims,
            max_dims=max_dims,
        ),
        label="input_shapes, result",
    )

    assert len(input_shapes) == num_shapes
    assert result == _broadcast_shapes(base_shape, *input_shapes)
    for shape in input_shapes:
        assert all(i == 1 for i in shape[-len(base_shape) :])


@given(
    num_shapes=st.integers(1, 3),
    base_shape=xps.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=5),
    max_dims=st.integers(0, 4),
    data=st.data(),
)
@settings(deadline=None)
def test_mutually_broadcastable_shapes_can_generate_arbitrary_ndims(
    num_shapes, base_shape, max_dims, data
):
    # ensures that each generated shape can possess any length in [min_dims, max_dims]
    desired_ndims = data.draw(
        st.lists(st.integers(0, max_dims), min_size=num_shapes, max_size=num_shapes),
        label="desired_ndims",
    )
    min_dims = data.draw(
        st.one_of(st.none(), st.integers(0, min(desired_ndims))), label="min_dims"
    )
    # check default arg behavior too
    kwargs = {"min_dims": min_dims} if min_dims is not None else {}
    find_any(
        xps.mutually_broadcastable_shapes(
            num_shapes=num_shapes,
            base_shape=base_shape,
            min_side=0,
            max_dims=max_dims,
            **kwargs,
        ),
        lambda x: {len(s) for s in x.input_shapes} == set(desired_ndims),
        settings(max_examples=10 ** 6),
    )
