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

import math
from collections import defaultdict
from functools import update_wrapper, wraps
from numbers import Real
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from warnings import warn

from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.extra.__array_helpers import (
    Shape,
    array_shapes,
    basic_indices,
    broadcastable_shapes,
    mutually_broadcastable_shapes,
    valid_tuple_axes,
)
from hypothesis.extra.numpy import valid_tuple_axes
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.validation import (
    check_type,
    check_valid_bound,
    check_valid_integer,
    check_valid_interval,
)
from hypothesis.strategies._internal.strategies import check_strategy
from hypothesis.strategies._internal.utils import defines_strategy

__all__ = [
    "get_strategies_namespace",
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
]


INT_NAMES = ["int8", "int16", "int32", "int64"]
UINT_NAMES = ["uint8", "uint16", "uint32", "uint64"]
ALL_INT_NAMES = INT_NAMES + UINT_NAMES
FLOAT_NAMES = ["float32", "float64"]
NUMERIC_NAMES = ALL_INT_NAMES + FLOAT_NAMES
DTYPE_NAMES = ["bool"] + NUMERIC_NAMES


def infer_xp_is_compliant(xp):
    try:
        array = xp.zeros(1)
        array.__array_namespace__()
    except Exception:
        warn(
            f"Could not determine whether module {xp} is an Array API library",
            HypothesisWarning,
        )


def check_xp_attributes(xp: Any, attributes: List[str]) -> None:
    missing_attrs = [attr for attr in attributes if not hasattr(xp, attr)]
    if len(missing_attrs) > 0:
        f_attrs = ", ".join(missing_attrs)
        raise InvalidArgument(
            f"Array module {xp} does not have required attributes: {f_attrs}"
        )


def partition_attributes_and_stubs(
    xp: Any, attributes: Iterable[str]
) -> Tuple[List[Any], List[str]]:
    non_stubs = []
    stubs = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)

    return non_stubs, stubs


def warn_on_missing_dtypes(xp: Any, stubs: List[str]) -> None:
    f_stubs = ", ".join(stubs)
    warn(
        f"Array module {xp} does not have the following "
        f"dtypes in its namespace: {f_stubs}",
        HypothesisWarning,
    )


def find_castable_builtin_for_dtype(
    xp: Any, dtype: Type
) -> Type[Union[bool, int, float]]:
    """Returns builtin type which can have values that are castable to the given
    dtype, according to :xp-ref:`type promotion rules <type_promotion.html>`.

    For floating dtypes we always return ``float``, even though ``int`` is also castable.
    """
    stubs = []

    try:
        bool_dtype = xp.bool
        if dtype == bool_dtype:
            return bool
    except AttributeError:
        stubs.append("bool")

    int_dtypes, int_stubs = partition_attributes_and_stubs(xp, ALL_INT_NAMES)
    if dtype in int_dtypes:
        return int

    float_dtypes, float_stubs = partition_attributes_and_stubs(xp, FLOAT_NAMES)
    if dtype in float_dtypes:
        return float

    stubs.extend(int_stubs)
    stubs.extend(float_stubs)
    if len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)
    raise InvalidArgument("dtype {dtype} not recognised in {xp}")


def dtype_from_name(xp: Any, name: str) -> Type:
    if name in DTYPE_NAMES:
        try:
            return getattr(xp, name)
        except AttributeError as e:
            raise InvalidArgument(
                f"Array module {xp} does not have dtype {name} in its namespace"
            ) from e
    else:
        f_valid_dtypes = ", ".join(DTYPE_NAMES)
        raise InvalidArgument(
            f"{name} is not a valid Array API data type "
            f"(pick from: {f_valid_dtypes})"
        )


class PrettyArrayModule:
    def __init__(self, xp):
        self._xp = xp
        if hasattr(xp, "_xp"):
            raise NotImplementedError(f"Array module {xp} cannot have attribute _xp")

    def __getattr__(self, name):
        return getattr(self._xp, name)

    def __repr__(self):
        try:
            return self._xp.__name__
        except AttributeError:
            return repr(self._xp)

    def __str__(self):
        return repr(self)


def pretty_xp_repr(func: Callable) -> Callable:
    """Wraps array module so it will have a pretty repr() and str().

    This namely prevents returned strategies having an ugly repr by way of the
    the defines_strategy decorator, which wraps the strategy in a LazyStrategy.
    A nice side effect is errors and warnings are easier to write.

    If ``xp`` is already a PrettyArrayModule then this behaviour is skipped.
    This prevents wrapped modules being wrapped again, which would happen when
    using the decorated strategies in practice.
    """

    @wraps(func)
    def inner(xp, *args, **kwargs):
        if not isinstance(xp, PrettyArrayModule):
            xp = PrettyArrayModule(xp)
        return func(xp, *args, **kwargs)

    return inner


@pretty_xp_repr
@defines_strategy(force_reusable_values=True)
def from_dtype(
    xp: Any,
    dtype: Union[Type, str],
    *,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
) -> st.SearchStrategy[Union[bool, int, float]]:
    """Return a strategy for any value of the given dtype.

    Values generated are of the Python scalar which is
    :xp-ref:`promotable <type_promotion.html>` to ``dtype``, where the values do
    not exceed its bounds.

    * ``dtype`` may be a dtype object or the string name of a
      :xp-ref:`valid dtype <data_types.html>`.

    Compatible ``**kwargs`` are passed to the inferred strategy function for
    integers and floats.  This allows you to customise the min and max values,
    and exclude non-finite numbers. This is particularly useful when kwargs are
    passed through from :func:`arrays`, as it seamlessly handles the ``width``
    or other representable bounds for you.
    """
    infer_xp_is_compliant(xp)
    check_xp_attributes(xp, ["iinfo", "finfo"])

    if isinstance(dtype, str):
        dtype = dtype_from_name(xp, dtype)
    builtin = find_castable_builtin_for_dtype(xp, dtype)

    def check_valid_minmax(prefix, val, info_obj):
        name = f"{prefix}_value"
        check_valid_bound(val, name)
        if val < info_obj.min:
            raise InvalidArgument(
                f"dtype {dtype} requires {name}={val} to be at least {info_obj.min}"
            )
        elif val > info_obj.max:
            raise InvalidArgument(
                f"dtype {dtype} requires {name}={val} to be at most {info_obj.max}"
            )

    if builtin is bool:
        return st.booleans()
    elif builtin is int:
        iinfo = xp.iinfo(dtype)
        if min_value is None:
            min_value = iinfo.min
        if max_value is None:
            max_value = iinfo.max
        check_valid_integer(min_value, "min_value")
        check_valid_integer(max_value, "max_value")
        assert isinstance(min_value, int)
        assert isinstance(max_value, int)
        check_valid_minmax("min", min_value, iinfo)
        check_valid_minmax("max", max_value, iinfo)
        check_valid_interval(min_value, max_value, "min_value", "max_value")
        return st.integers(min_value=min_value, max_value=max_value)
    else:
        finfo = xp.finfo(dtype)
        kw = {}

        # Whilst we know the boundary values of float dtypes from finfo, we do
        # not assign them to the floats() strategy by default - passing min/max
        # values will modify test case reduction behaviour so that simple bugs
        # may become harder for users to identify. We plan to improve floats()
        # behaviour in https://github.com/HypothesisWorks/hypothesis/issues/2907.
        # Setting width should manage boundary values for us anyway.
        if min_value is not None:
            check_valid_bound(min_value, "min_value")
            assert isinstance(min_value, Real)
            check_valid_minmax("min", min_value, finfo)
            kw["min_value"] = min_value
        if max_value is not None:
            check_valid_bound(max_value, "max_value")
            assert isinstance(max_value, Real)
            check_valid_minmax("max", max_value, finfo)
            if min_value is not None:
                check_valid_interval(min_value, max_value, "min_value", "max_value")
            kw["max_value"] = max_value

        if allow_nan is not None:
            kw["allow_nan"] = allow_nan
        if allow_infinity is not None:
            kw["allow_infinity"] = allow_infinity
        if exclude_min is not None:
            kw["exclude_min"] = exclude_min
        if exclude_max is not None:
            kw["exclude_max"] = exclude_max

        return st.floats(width=finfo.bits, **kw)


class ArrayStrategy(st.SearchStrategy):
    def __init__(self, xp, elements_strategy, dtype, shape, fill, unique):
        self.xp = xp
        self.elements_strategy = elements_strategy
        self.dtype = dtype
        self.shape = shape
        self.fill = fill
        self.unique = unique
        self.array_size = math.prod(shape)
        self.builtin = find_castable_builtin_for_dtype(xp, dtype)

    def set_value(self, result, i, val, strategy=None):
        strategy = strategy or self.elements_strategy
        try:
            result[i] = val
        except TypeError as e:
            raise InvalidArgument(
                f"Could not add generated array element {val!r} "
                f"of dtype {type(val)} to array of dtype {result.dtype}."
            ) from e
        self.check_set_value(val, result[i], strategy)

    def check_set_value(self, val, val_0d, strategy):
        if self.builtin is bool:
            finite = True
        else:
            finite = self.xp.isfinite(val_0d)
        if finite and self.builtin(val_0d) != val:
            raise InvalidArgument(
                f"Generated array element {val!r} from strategy {strategy} "
                f"cannot be represented as dtype {self.dtype}. "
                f"Array module {self.xp.__name__} instead "
                f"represents the element as {val_0d!r}. "
                "Consider using a more precise elements strategy, "
                "for example passing the width argument to floats()."
            )

    def do_draw(self, data):
        if 0 in self.shape:
            return self.xp.zeros(self.shape, dtype=self.dtype)

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.

            # This could legitimately be a xp.empty, but the performance gains
            # for that are likely marginal, so there's really not much point
            # risking undefined behaviour shenanigans.
            result = self.xp.zeros(self.array_size, dtype=self.dtype)

            if self.unique:
                seen = set()
                elements = cu.many(
                    data,
                    min_size=self.array_size,
                    max_size=self.array_size,
                    average_size=self.array_size,
                )
                i = 0
                while elements.more():
                    val = data.draw(self.elements_strategy)
                    if val in seen:
                        elements.reject()
                    else:
                        seen.add(val)
                        self.set_value(result, i, val)
                        i += 1
            else:
                for i in range(self.array_size):
                    val = data.draw(self.elements_strategy)
                    self.set_value(result, i, val)
        else:
            # We draw arrays as "sparse with an offset". We assume not every
            # element will be assigned and so first draw a single value from our
            # fill strategy to create a full array. We then draw a collection of
            # index assignments within the array and assign fresh values from
            # our elements strategy to those indices.

            fill_val = data.draw(self.fill)
            try:
                result = self.xp.full(self.array_size, fill_val, dtype=self.dtype)
            except Exception as e:
                raise InvalidArgument(
                    f"Could not create full array of dtype {self.dtype} "
                    f"with fill value {fill_val!r}"
                ) from e
            sample = result[0]
            self.check_set_value(fill_val, sample, strategy=self.fill)
            if self.unique and not self.xp.all(self.xp.isnan(result)):
                raise InvalidArgument(
                    f"Array module {self.xp.__name__} did not recognise fill "
                    f"value {fill_val!r} as NaN - instead got {sample!r}. "
                    "Cannot fill unique array with non-NaN values."
                )

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It
                # just grows reasonably quickly but sublinearly, and for small
                # arrays it represents a decent fraction of the array size.
                average_size=math.sqrt(self.array_size),
            )

            index_set = defaultdict(bool)
            seen = set()

            while elements.more():
                i = cu.integer_range(data, 0, self.array_size - 1)
                if index_set[i]:
                    elements.reject()
                    continue
                val = data.draw(self.elements_strategy)
                if self.unique:
                    if val in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(val)
                self.set_value(result, i, val)
                index_set[i] = True

        result = self.xp.reshape(result, self.shape)

        return result


@pretty_xp_repr
@defines_strategy(force_reusable_values=True)
def arrays(
    xp: Any,
    dtype: Union[Type, str, st.SearchStrategy[Type], st.SearchStrategy[str]],
    shape: Union[int, Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[st.SearchStrategy] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> st.SearchStrategy:
    """Returns a strategy for :xp-ref:`arrays <array_object.html>`.

    * ``dtype`` may be a :xp-ref:`valid dtype <data_types.html>` object or name,
      or a strategy that generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a strategy
      that generates such values.
    * ``elements`` is a strategy for values to put in the array. If ``None``
      then a suitable value will be inferred based on the dtype, which may give
      any legal value (including e.g. NaN for floats). If a mapping, it will be
      passed as ``**kwargs`` to ``from_dtype()`` when inferring based on the dtype.
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If ``None``, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling behaviour will be
      disabled entirely and every element will be generated independently.
    * ``unique`` specifies if the elements of the array should all be distinct
      from one another. Note that in this case multiple NaN values may still be
      allowed. If fill is also set, the only valid values for fill to return are
      NaN values.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> from numpy import array_api as xp
      >>> arrays(xp, xp.int8, (2, 3)).example()
      Array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)

    Specifying element boundaries by a :obj:`python:dict` of the kwargs to pass
    to :func:`from_dtype` will ensure ``dtype`` bounds will be respected.

    .. code-block:: pycon

      >>> arrays(xp, xp.int8, 3, elements={"min_value": 10}).example()
      Array([125, 13, 79], dtype=int8)

    Refer to :doc:`What you can generate and how <data>` for passing
    your own elements strategy.

    .. code-block:: pycon

      >>> arrays(xp, xp.float32, 3, elements=floats(0, 1, width=32)).example()
      Array([ 0.88974794,  0.77387938,  0.1977879 ], dtype=float32)

    Array values are generated in two parts:

    1. A single value is drawn from the fill strategy and is used to create a
       filled array.
    2. Some subset of the coordinates of the array are populated with a value
       drawn from the elements strategy (or its inferred form).

    You can set ``fill`` to :func:`~hypothesis.strategies.nothing` if you want
    to disable this behaviour and draw a value for every element.

    By default ``arrays`` will attempt to infer the correct fill behaviour: if
    ``unique`` is also ``True``, no filling will occur. Otherwise, if it looks
    safe to reuse the values of elements across multiple coordinates (this will
    be the case for any inferred strategy, and for most of the builtins, but is
    not the case for mutable values or strategies built with flatmap, map,
    composite, etc.) then it will use the elements strategy as the fill, else it
    will default to having no fill.

    Having a fill helps Hypothesis craft high quality examples, but its
    main importance is when the array generated is large: Hypothesis is
    primarily designed around testing small examples. If you have arrays with
    hundreds or more elements, having a fill value is essential if you want
    your tests to run in reasonable time.
    """

    infer_xp_is_compliant(xp)
    check_xp_attributes(xp, ["zeros", "full", "all", "isnan", "isfinite", "reshape"])

    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(xp, d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(xp, dtype, s, elements=elements, fill=fill, unique=unique)
        )

    if isinstance(dtype, str):
        dtype = dtype_from_name(xp, dtype)

    if isinstance(shape, int):
        shape = (shape,)
    if not all(isinstance(s, int) for s in shape):
        raise InvalidArgument(
            f"Array shape must be integer in each dimension, provided shape was {shape}"
        )

    if elements is None:
        elements = from_dtype(xp, dtype)
    elif isinstance(elements, Mapping):
        elements = from_dtype(xp, dtype, **elements)
    check_strategy(elements, "elements")

    if fill is None:
        assert isinstance(elements, st.SearchStrategy)  # for mypy
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    check_strategy(fill, "fill")

    return ArrayStrategy(xp, elements, dtype, shape, fill, unique)


def check_dtypes(xp: Any, dtypes: List[Type], stubs: List[str]) -> None:
    if len(dtypes) == 0:
        f_stubs = ", ".join(stubs)
        raise InvalidArgument(
            f"Array module {xp} does not have the following "
            f"required dtypes in its namespace: {f_stubs}"
        )
    elif len(stubs) > 0:
        warn_on_missing_dtypes(xp, stubs)


@pretty_xp_repr
@defines_strategy()
def scalar_dtypes(xp: Any) -> st.SearchStrategy[Type]:
    """Return a strategy for all :xp-ref:`valid dtype <data_types.html>` objects."""
    infer_xp_is_compliant(xp)
    return st.one_of(boolean_dtypes(xp), numeric_dtypes(xp))


@pretty_xp_repr
@defines_strategy()
def boolean_dtypes(xp: Any) -> st.SearchStrategy[Type]:
    infer_xp_is_compliant(xp)
    try:
        return st.just(xp.bool)
    except AttributeError:
        raise InvalidArgument(
            f"Array module {xp} does not have a bool dtype in its namespace"
        ) from None


@pretty_xp_repr
@defines_strategy()
def numeric_dtypes(xp: Any) -> st.SearchStrategy[Type]:
    """Return a strategy for all numeric dtype objects."""
    infer_xp_is_compliant(xp)
    return st.one_of(
        integer_dtypes(xp),
        unsigned_integer_dtypes(xp),
        floating_dtypes(xp),
    )


def check_valid_sizes(
    category: str, sizes: Sequence[int], valid_sizes: Sequence[int]
) -> None:
    invalid_sizes = []
    for size in sizes:
        if size not in valid_sizes:
            invalid_sizes.append(size)
    if len(invalid_sizes) > 0:
        f_valid_sizes = ", ".join(str(s) for s in valid_sizes)
        f_invalid_sizes = ", ".join(str(s) for s in invalid_sizes)
        raise InvalidArgument(
            f"The following sizes are not valid for {category} dtypes: "
            f"{f_invalid_sizes} (valid sizes: {f_valid_sizes})"
        )


def numeric_dtype_names(base_name: str, sizes: Sequence[int]) -> Iterator[str]:
    for size in sizes:
        yield f"{base_name}{size}"


@pretty_xp_repr
@defines_strategy()
def integer_dtypes(
    xp: Any, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type]:
    """Return a strategy for signed integer dtype objects.

    ``sizes`` contains the signed integer sizes in bits, defaulting to
    ``(8, 16, 32, 64)`` which covers all valid sizes.
    """
    infer_xp_is_compliant(xp)
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))
    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("int", sizes)
    )
    check_dtypes(xp, dtypes, stubs)
    return st.sampled_from(dtypes)


@pretty_xp_repr
@defines_strategy()
def unsigned_integer_dtypes(
    xp: Any, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
) -> st.SearchStrategy[Type]:
    """Return a strategy for unsigned integer dtype objects.

    ``sizes`` contains the unsigned integer sizes in bits, defaulting to
    ``(8, 16, 32, 64)`` which covers all valid sizes.
    """
    infer_xp_is_compliant(xp)

    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (8, 16, 32, 64))

    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("uint", sizes)
    )
    check_dtypes(xp, dtypes, stubs)

    return st.sampled_from(dtypes)


@pretty_xp_repr
@defines_strategy()
def floating_dtypes(
    xp: Any, *, sizes: Union[int, Sequence[int]] = (32, 64)
) -> st.SearchStrategy[Type]:
    """Return a strategy for floating-point dtype objects.

    ``sizes`` contains the floating-point sizes in bits, defaulting to
    ``(32, 64)`` which covers all valid sizes.
    """

    infer_xp_is_compliant(xp)
    if isinstance(sizes, int):
        sizes = (sizes,)
    check_valid_sizes("int", sizes, (32, 64))
    dtypes, stubs = partition_attributes_and_stubs(
        xp, numeric_dtype_names("float", sizes)
    )
    check_dtypes(xp, dtypes, stubs)
    return st.sampled_from(dtypes)


valid_tuple_axes.__doc__ = f"""
    Return a strategy for permissible tuple-values for the ``axis``
    argument in Array API sequential methods e.g. ``sum``, given the specified
    dimensionality.

    {valid_tuple_axes.__doc__}
    """

# TODO: mutually_broadcastable_shapes exposes sig stuff

# TODO: min_dims defaults to 1, prevent indices for 0d shapes
indices = basic_indices
indices.__doc__ = f"""
    Return a strategy for :xp-ref:`valid indices <indexing.html>` of
    arrays with the specified shape.

    {basic_indices.__doc__}
    """


@pretty_xp_repr
def get_strategies_namespace(xp: Any) -> SimpleNamespace:
    """Creates a strategies namespace for the given array module.

    * ``xp`` is the Array API library to automatically pass to the namespaced methods.

    A :obj:`python:types.SimpleNamespace` is returned which contains all the
    strategy methods in this module but without requiring the ``xp`` argument.

    Creating and using a strategies namespace for NumPy's Array API
    implementation would go like this:

    .. code-block:: pycon

      >>> from numpy import array_api as xp
      >>> xps = get_strategies_namespace(xp)
      >>> x = xps.arrays(xp.int8, (2, 3)).example()
      >>> x
      Array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)
      >>> x.__array_namespace__() is xp
      True

    """
    infer_xp_is_compliant(xp)

    return SimpleNamespace(
        from_dtype=update_wrapper(
            lambda *a, **kw: from_dtype(xp, *a, **kw), from_dtype
        ),
        arrays=update_wrapper(lambda *a, **kw: arrays(xp, *a, **kw), arrays),
        array_shapes=array_shapes,
        scalar_dtypes=update_wrapper(
            lambda *a, **kw: scalar_dtypes(xp, *a, **kw), scalar_dtypes
        ),
        boolean_dtypes=update_wrapper(
            lambda *a, **kw: boolean_dtypes(xp, *a, **kw), boolean_dtypes
        ),
        numeric_dtypes=update_wrapper(
            lambda *a, **kw: numeric_dtypes(xp, *a, **kw), numeric_dtypes
        ),
        integer_dtypes=update_wrapper(
            lambda *a, **kw: integer_dtypes(xp, *a, **kw), integer_dtypes
        ),
        unsigned_integer_dtypes=update_wrapper(
            lambda *a, **kw: unsigned_integer_dtypes(xp, *a, **kw),
            unsigned_integer_dtypes,
        ),
        floating_dtypes=update_wrapper(
            lambda *a, **kw: floating_dtypes(xp, *a, **kw), floating_dtypes
        ),
        valid_tuple_axes=valid_tuple_axes,
        broadcastable_shapes=broadcastable_shapes,
        mutually_broadcastable_shapes=mutually_broadcastable_shapes,
        indices=indices,
    )
