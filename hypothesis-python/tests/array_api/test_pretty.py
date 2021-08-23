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

from tests.array_api.xputils import xp, xps

pytestmark = [pytest.mark.mockable_xp]


@pytest.mark.parametrize(
    "name",
    [
        "from_dtype",
        "arrays",
        "array_shapes",
        "scalar_dtypes",
        "boolean_dtypes",
        "numeric_dtypes",
        "integer_dtypes",
        "unsigned_integer_dtypes",
        "floating_dtypes",
        "valid_tuple_axes",
        "broadcastable_shapes",
        "mutually_broadcastable_shapes",
        "indices",
    ],
)
def test_namespaced_methods_wrapped(name):
    """Namespaced strategies have readable method names, even if they are lambdas."""
    func = getattr(xps, name)
    assert func.__name__ == name


@pytest.mark.parametrize(
    "name, strat",
    [
        ("from_dtype", xps.from_dtype(xp.int8)),
        ("arrays", xps.arrays(xp.int8, 5)),
        ("scalar_dtypes", xps.scalar_dtypes()),
        ("boolean_dtypes", xps.boolean_dtypes()),
        ("numeric_dtypes", xps.numeric_dtypes()),
        ("integer_dtypes", xps.integer_dtypes()),
        ("unsigned_integer_dtypes", xps.unsigned_integer_dtypes()),
        ("floating_dtypes", xps.floating_dtypes()),
    ],
)
def test_xp_strategies_pretty_repr(name, strat):
    """Strategies that take xp use its __name__ for their own repr."""
    assert repr(strat).startswith(name), f"{name} not in strat repr"
    assert len(repr(strat)) < 100, "strat repr looks too long"
    assert xp.__name__ in repr(strat), f"{xp.__name__} not in strat repr"
