import pytest

from hypothesis.extra.array_api import (
    arrays,
    boolean_dtypes,
    floating_dtypes,
    from_dtype,
    integer_dtypes,
    numeric_dtypes,
    scalar_dtypes,
    unsigned_integer_dtypes,
)

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
        # Generic strategies
        ("from_dtype", from_dtype(xp, xp.int8)),
        ("arrays", arrays(xp, xp.int8, 5)),
        ("scalar_dtypes", scalar_dtypes(xp)),
        ("boolean_dtypes", boolean_dtypes(xp)),
        ("numeric_dtypes", numeric_dtypes(xp)),
        ("integer_dtypes", integer_dtypes(xp)),
        ("unsigned_integer_dtypes", unsigned_integer_dtypes(xp)),
        ("floating_dtypes", floating_dtypes(xp)),
        # Namespaced strategies
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
