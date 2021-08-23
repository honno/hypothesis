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

import pytest

from hypothesis import given, strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.extra.array_api import (
    DTYPE_NAMES,
    FLOAT_NAMES,
    INT_NAMES,
    UINT_NAMES,
    arrays,
    boolean_dtypes,
    floating_dtypes,
    from_dtype,
    integer_dtypes,
    numeric_dtypes,
    scalar_dtypes,
    unsigned_integer_dtypes,
)

from tests.array_api.xputils import MOCK_NAME, create_array_module

noncompliant_xp = create_array_module()


@pytest.mark.parametrize(
    "func, args",
    [
        (from_dtype, ["int8"]),
        (arrays, ["int8", 5]),
        (scalar_dtypes, []),
        (boolean_dtypes, []),
        (numeric_dtypes, []),
        (integer_dtypes, []),
        (unsigned_integer_dtypes, []),
        (floating_dtypes, []),
    ],
)
def test_warning_on_noncompliant_arrays(func, args):
    """Strategies using array modules with non-compliant array objects execute
    with a warning"""
    with pytest.warns(HypothesisWarning, match=f"determine.*{MOCK_NAME}.*Array API"):
        func(noncompliant_xp, *args).example()


@pytest.mark.filterwarnings(f"ignore:.*determine.*{MOCK_NAME}.*Array API.*")
@pytest.mark.parametrize(
    "func, args, attr",
    [(from_dtype, ["int8"], "iinfo"), (arrays, ["int8", 5], "full")],
)
def test_error_on_missing_attr(func, args, attr):
    """Strategies raise helpful error when using array modules that lack
    required attributes."""
    xp = create_array_module(exclude=(attr,))
    with pytest.raises(InvalidArgument, match=f"{MOCK_NAME}.*required.*{attr}"):
        func(xp, *args).example()


dtypeless_xp = create_array_module(exclude=tuple(DTYPE_NAMES))


@pytest.mark.filterwarnings(f"ignore:.*determine.*{MOCK_NAME}.*Array API.*")
@pytest.mark.parametrize(
    "func",
    [
        scalar_dtypes,
        boolean_dtypes,
        numeric_dtypes,
        integer_dtypes,
        unsigned_integer_dtypes,
        floating_dtypes,
    ],
)
def test_error_on_missing_dtypes(func):
    """Strategies raise helpful error when using array modules that lack
    required dtypes."""
    with pytest.raises(InvalidArgument, match=f"{MOCK_NAME}.*dtype.*namespace"):
        func(dtypeless_xp).example()


@pytest.mark.filterwarnings(f"ignore:.*determine.*{MOCK_NAME}.*Array API.*")
@pytest.mark.parametrize(
    "func, keep_anys",
    [
        (scalar_dtypes, [INT_NAMES, UINT_NAMES, FLOAT_NAMES]),
        (numeric_dtypes, [INT_NAMES, UINT_NAMES, FLOAT_NAMES]),
        (integer_dtypes, [INT_NAMES]),
        (unsigned_integer_dtypes, [UINT_NAMES]),
        (floating_dtypes, [FLOAT_NAMES]),
    ],
)
@given(st.data())
def test_warning_on_partial_dtypes(func, keep_anys, data):
    """Strategies using array modules with at least one of a dtype in the
    necessary category/categories execute with a warning.
    """
    exclude = []
    for keep_any in keep_anys:
        exclude.extend(
            data.draw(
                st.lists(
                    st.sampled_from(keep_any),
                    min_size=1,
                    max_size=len(keep_any) - 1,
                    unique=True,
                )
            )
        )
    xp = create_array_module(exclude=tuple(exclude))
    with pytest.warns(HypothesisWarning, match=f"{MOCK_NAME}.*dtype.*namespace"):
        data.draw(func(xp))
