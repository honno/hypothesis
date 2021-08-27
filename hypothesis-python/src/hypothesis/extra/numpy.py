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
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from hypothesis import strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.extra import _array_helpers
from hypothesis.extra._array_helpers import (
    BasicIndex,
    BroadcastableShapes,
    Shape,
    check_argument,
    order_check,
)
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.coverage import check_function
from hypothesis.internal.reflection import proxies
from hypothesis.internal.validation import check_type
from hypothesis.strategies._internal.strategies import T, check_strategy
from hypothesis.strategies._internal.utils import defines_strategy

__all__ = [
    "Shape",
    "BroadcastableShapes",
    "BasicIndex",
    "TIME_RESOLUTIONS",
    "from_dtype",
    "arrays",
    "array_shapes",
    "scalar_dtypes",
    "boolean_dtypes",
    "unsigned_integer_dtypes",
    "integer_dtypes",
    "floating_dtypes",
    "complex_number_dtypes",
    "datetime64_dtypes",
    "timedelta64_dtypes",
    "byte_string_dtypes",
    "unicode_string_dtypes",
    "array_dtypes",
    "nested_dtypes",
    "valid_tuple_axes",
    "broadcastable_shapes",
    "mutually_broadcastable_shapes",
    "basic_indices",
    "integer_array_indices",
]

TIME_RESOLUTIONS = tuple("Y  M  D  h  m  s  ms  us  ns  ps  fs  as".split())


@defines_strategy(force_reusable_values=True)
def from_dtype(
    dtype: np.dtype,
    *,
    alphabet: Optional[st.SearchStrategy[str]] = None,
    min_size: int = 0,
    max_size: Optional[int] = None,
    min_value: Union[int, float, None] = None,
    max_value: Union[int, float, None] = None,
    allow_nan: Optional[bool] = None,
    allow_infinity: Optional[bool] = None,
    exclude_min: Optional[bool] = None,
    exclude_max: Optional[bool] = None,
) -> st.SearchStrategy[Any]:
    """Creates a strategy which can generate any value of the given dtype.

    Compatible ``**kwargs`` are passed to the inferred strategy function for
    integers, floats, and strings.  This allows you to customise the min and max
    values, control the length or contents of strings, or exclude non-finite
    numbers.  This is particularly useful when kwargs are passed through from
    :func:`arrays` which allow a variety of numeric dtypes, as it seamlessly
    handles the ``width`` or representable bounds for you.  See :issue:`2552`
    for more detail.
    """
    check_type(np.dtype, dtype, "dtype")
    kwargs = {k: v for k, v in locals().items() if k != "dtype" and v is not None}

    # Compound datatypes, eg 'f4,f4,f4'
    if dtype.names is not None:
        # mapping np.void.type over a strategy is nonsense, so return now.
        subs = [from_dtype(dtype.fields[name][0], **kwargs) for name in dtype.names]
        return st.tuples(*subs)

    # Subarray datatypes, eg '(2, 3)i4'
    if dtype.subdtype is not None:
        subtype, shape = dtype.subdtype
        return arrays(subtype, shape, elements=kwargs)

    def compat_kw(*args, **kw):
        """Update default args to the strategy with user-supplied keyword args."""
        assert {"min_value", "max_value", "max_size"}.issuperset(kw)
        for key in set(kwargs).intersection(kw):
            msg = f"dtype {dtype!r} requires {key}={kwargs[key]!r} to be %s {kw[key]!r}"
            if kw[key] is not None:
                if key.startswith("min_") and kw[key] > kwargs[key]:
                    raise InvalidArgument(msg % ("at least",))
                elif key.startswith("max_") and kw[key] < kwargs[key]:
                    raise InvalidArgument(msg % ("at most",))
        kw.update({k: v for k, v in kwargs.items() if k in args or k in kw})
        return kw

    # Scalar datatypes
    if dtype.kind == "b":
        result: st.SearchStrategy[Any] = st.booleans()
    elif dtype.kind == "f":
        result = st.floats(
            width=8 * dtype.itemsize,
            **compat_kw(
                "min_value",
                "max_value",
                "allow_nan",
                "allow_infinity",
                "exclude_min",
                "exclude_max",
            ),
        )
    elif dtype.kind == "c":
        # If anyone wants to add a `width` argument to `complex_numbers()`, we would
        # accept a pull request and add passthrough support for magnitude bounds,
        # but it's a low priority otherwise.
        if dtype.itemsize == 8:
            float32 = st.floats(width=32, **compat_kw("allow_nan", "allow_infinity"))
            result = st.builds(complex, float32, float32)
        else:
            result = st.complex_numbers(**compat_kw("allow_nan", "allow_infinity"))
    elif dtype.kind in ("S", "a"):
        # Numpy strings are null-terminated; only allow round-trippable values.
        # `itemsize == 0` means 'fixed length determined at array creation'
        max_size = dtype.itemsize or None
        result = st.binary(**compat_kw("min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != b"\0"
        )
    elif dtype.kind == "u":
        kw = compat_kw(min_value=0, max_value=2 ** (8 * dtype.itemsize) - 1)
        result = st.integers(**kw)
    elif dtype.kind == "i":
        overflow = 2 ** (8 * dtype.itemsize - 1)
        result = st.integers(**compat_kw(min_value=-overflow, max_value=overflow - 1))
    elif dtype.kind == "U":
        # Encoded in UTF-32 (four bytes/codepoint) and null-terminated
        max_size = (dtype.itemsize or 0) // 4 or None
        result = st.text(**compat_kw("alphabet", "min_size", max_size=max_size)).filter(
            lambda b: b[-1:] != "\0"
        )
    elif dtype.kind in ("m", "M"):
        if "[" in dtype.str:
            res = st.just(dtype.str.split("[")[-1][:-1])
        else:
            # Note that this case isn't valid to pass to arrays(), but we support
            # it here because we'd have to guard against equivalents in arrays()
            # regardless and drawing scalars is a valid use-case.
            res = st.sampled_from(TIME_RESOLUTIONS)
        result = st.builds(dtype.type, st.integers(-(2 ** 63), 2 ** 63 - 1), res)
    else:
        raise InvalidArgument(f"No strategy inference for {dtype}")
    return result.map(dtype.type)


class ArrayStrategy(st.SearchStrategy):
    def __init__(self, element_strategy, shape, dtype, fill, unique):
        self.shape = tuple(shape)
        self.fill = fill
        self.array_size = int(np.prod(shape))
        self.dtype = dtype
        self.element_strategy = element_strategy
        self.unique = unique
        self._check_elements = dtype.kind not in ("O", "V")

    def set_element(self, data, result, idx, strategy=None):
        strategy = strategy or self.element_strategy
        val = data.draw(strategy)
        try:
            result[idx] = val
        except TypeError as err:
            raise InvalidArgument(
                f"Could not add element={val!r} of {val.dtype!r} to array of "
                f"{result.dtype!r} - possible mismatch of time units in dtypes?"
            ) from err
        if self._check_elements and val != result[idx] and val == val:
            raise InvalidArgument(
                "Generated array element %r from %r cannot be represented as "
                "dtype %r - instead it becomes %r (type %r).  Consider using a more "
                "precise strategy, for example passing the `width` argument to "
                "`floats()`."
                % (val, strategy, self.dtype, result[idx], type(result[idx]))
            )

    def do_draw(self, data):
        if 0 in self.shape:
            return np.zeros(dtype=self.dtype, shape=self.shape)

        # Because Numpy allocates memory for strings at array creation, if we have
        # an unsized string dtype we'll fill an object array and then cast it back.
        unsized_string_dtype = (
            self.dtype.kind in ("S", "a", "U") and self.dtype.itemsize == 0
        )

        # This could legitimately be a np.empty, but the performance gains for
        # that would be so marginal that there's really not much point risking
        # undefined behaviour shenanigans.
        result = np.zeros(
            shape=self.array_size, dtype=object if unsized_string_dtype else self.dtype
        )

        if self.fill.is_empty:
            # We have no fill value (either because the user explicitly
            # disabled it or because the default behaviour was used and our
            # elements strategy does not produce reusable values), so we must
            # generate a fully dense array with a freshly drawn value for each
            # entry.
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
                    # We assign first because this means we check for
                    # uniqueness after numpy has converted it to the relevant
                    # type for us. Because we don't increment the counter on
                    # a duplicate we will overwrite it on the next draw.
                    self.set_element(data, result, i)
                    if result[i] not in seen:
                        seen.add(result[i])
                        i += 1
                    else:
                        elements.reject()
            else:
                for i in range(len(result)):
                    self.set_element(data, result, i)
        else:
            # We draw numpy arrays as "sparse with an offset". We draw a
            # collection of index assignments within the array and assign
            # fresh values from our elements strategy to those indices. If at
            # the end we have not assigned every element then we draw a single
            # value from our fill strategy and use that to populate the
            # remaining positions with that strategy.

            elements = cu.many(
                data,
                min_size=0,
                max_size=self.array_size,
                # sqrt isn't chosen for any particularly principled reason. It
                # just grows reasonably quickly but sublinearly, and for small
                # arrays it represents a decent fraction of the array size.
                average_size=math.sqrt(self.array_size),
            )

            needs_fill = np.full(self.array_size, True)
            seen = set()

            while elements.more():
                i = cu.integer_range(data, 0, self.array_size - 1)
                if not needs_fill[i]:
                    elements.reject()
                    continue
                self.set_element(data, result, i)
                if self.unique:
                    if result[i] in seen:
                        elements.reject()
                        continue
                    else:
                        seen.add(result[i])
                needs_fill[i] = False
            if needs_fill.any():
                # We didn't fill all of the indices in the early loop, so we
                # put a fill value into the rest.

                # We have to do this hilarious little song and dance to work
                # around numpy's special handling of iterable values. If the
                # value here were e.g. a tuple then neither array creation
                # nor putmask would do the right thing. But by creating an
                # array of size one and then assigning the fill value as a
                # single element, we both get an array with the right value in
                # it and putmask will do the right thing by repeating the
                # values of the array across the mask.
                one_element = np.zeros(
                    shape=1, dtype=object if unsized_string_dtype else self.dtype
                )
                self.set_element(data, one_element, 0, self.fill)
                if unsized_string_dtype:
                    one_element = one_element.astype(self.dtype)
                fill_value = one_element[0]
                if self.unique:
                    try:
                        is_nan = np.isnan(fill_value)
                    except TypeError:
                        is_nan = False

                    if not is_nan:
                        raise InvalidArgument(
                            f"Cannot fill unique array with non-NaN value {fill_value!r}"
                        )

                np.putmask(result, needs_fill, one_element)

        if unsized_string_dtype:
            out = result.astype(self.dtype)
            mismatch = out != result
            if mismatch.any():
                raise InvalidArgument(
                    "Array elements %r cannot be represented as dtype %r - instead "
                    "they becomes %r.  Use a more precise strategy, e.g. without "
                    "trailing null bytes, as this will be an error future versions."
                    % (result[mismatch], self.dtype, out[mismatch])
                )
            result = out

        result = result.reshape(self.shape).copy()

        assert result.base is None

        return result


@check_function
def fill_for(elements, unique, fill, name=""):
    if fill is None:
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    else:
        check_strategy(fill, f"{name}.fill" if name else "fill")
    return fill


@defines_strategy(force_reusable_values=True)
def arrays(
    dtype: Any,
    shape: Union[int, st.SearchStrategy[int], Shape, st.SearchStrategy[Shape]],
    *,
    elements: Optional[Union[st.SearchStrategy, Mapping[str, Any]]] = None,
    fill: Optional[st.SearchStrategy[Any]] = None,
    unique: bool = False,
) -> st.SearchStrategy[np.ndarray]:
    r"""Returns a strategy for generating :class:`numpy:numpy.ndarray`\ s.

    * ``dtype`` may be any valid input to :class:`~numpy:numpy.dtype`
      (this includes :class:`~numpy:numpy.dtype` objects), or a strategy that
      generates such values.
    * ``shape`` may be an integer >= 0, a tuple of such integers, or a
      strategy that generates such values.
    * ``elements`` is a strategy for generating values to put in the array.
      If it is None a suitable value will be inferred based on the dtype,
      which may give any legal value (including eg ``NaN`` for floats).
      If a mapping, it will be passed as ``**kwargs`` to ``from_dtype()``
    * ``fill`` is a strategy that may be used to generate a single background
      value for the array. If None, a suitable default will be inferred
      based on the other arguments. If set to
      :func:`~hypothesis.strategies.nothing` then filling
      behaviour will be disabled entirely and every element will be generated
      independently.
    * ``unique`` specifies if the elements of the array should all be
      distinct from one another. Note that in this case multiple NaN values
      may still be allowed. If fill is also set, the only valid values for
      it to return are NaN values (anything for which :obj:`numpy:numpy.isnan`
      returns True. So e.g. for complex numbers (nan+1j) is also a valid fill).
      Note that if unique is set to True the generated values must be hashable.

    Arrays of specified ``dtype`` and ``shape`` are generated for example
    like this:

    .. code-block:: pycon

      >>> import numpy as np
      >>> arrays(np.int8, (2, 3)).example()
      array([[-8,  6,  3],
             [-6,  4,  6]], dtype=int8)

    - See :doc:`What you can generate and how <data>`.

    .. code-block:: pycon

      >>> import numpy as np
      >>> from hypothesis.strategies import floats
      >>> arrays(np.float, 3, elements=floats(0, 1)).example()
      array([ 0.88974794,  0.77387938,  0.1977879 ])

    Array values are generated in two parts:

    1. Some subset of the coordinates of the array are populated with a value
       drawn from the elements strategy (or its inferred form).
    2. If any coordinates were not assigned in the previous step, a single
       value is drawn from the fill strategy and is assigned to all remaining
       places.

    You can set fill to :func:`~hypothesis.strategies.nothing` if you want to
    disable this behaviour and draw a value for every element.

    If fill is set to None then it will attempt to infer the correct behaviour
    automatically: If unique is True, no filling will occur by default.
    Otherwise, if it looks safe to reuse the values of elements across
    multiple coordinates (this will be the case for any inferred strategy, and
    for most of the builtins, but is not the case for mutable values or
    strategies built with flatmap, map, composite, etc) then it will use the
    elements strategy as the fill, else it will default to having no fill.

    Having a fill helps Hypothesis craft high quality examples, but its
    main importance is when the array generated is large: Hypothesis is
    primarily designed around testing small examples. If you have arrays with
    hundreds or more elements, having a fill value is essential if you want
    your tests to run in reasonable time.
    """
    # We support passing strategies as arguments for convenience, or at least
    # for legacy reasons, but don't want to pay the perf cost of a composite
    # strategy (i.e. repeated argument handling and validation) when it's not
    # needed.  So we get the best of both worlds by recursing with flatmap,
    # but only when it's actually needed.
    if isinstance(dtype, st.SearchStrategy):
        return dtype.flatmap(
            lambda d: arrays(d, shape, elements=elements, fill=fill, unique=unique)
        )
    if isinstance(shape, st.SearchStrategy):
        return shape.flatmap(
            lambda s: arrays(dtype, s, elements=elements, fill=fill, unique=unique)
        )
    # From here on, we're only dealing with values and it's relatively simple.
    dtype = np.dtype(dtype)
    if elements is None or isinstance(elements, Mapping):
        if dtype.kind in ("m", "M") and "[" not in dtype.str:
            # For datetime and timedelta dtypes, we have a tricky situation -
            # because they *may or may not* specify a unit as part of the dtype.
            # If not, we flatmap over the various resolutions so that array
            # elements have consistent units but units may vary between arrays.
            return (
                st.sampled_from(TIME_RESOLUTIONS)
                .map((dtype.str + "[{}]").format)
                .flatmap(lambda d: arrays(d, shape=shape, fill=fill, unique=unique))
            )
        elements = from_dtype(dtype, **(elements or {}))
    check_strategy(elements, "elements")
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    check_argument(
        all(isinstance(s, int) for s in shape),
        "Array shape must be integer in each dimension, provided shape was {}",
        shape,
    )
    fill = fill_for(elements=elements, unique=unique, fill=fill)
    return ArrayStrategy(elements, shape, dtype, fill, unique)


array_shapes = _array_helpers.array_shapes


@defines_strategy()
def scalar_dtypes() -> st.SearchStrategy[np.dtype]:
    """Return a strategy that can return any non-flexible scalar dtype."""
    return st.one_of(
        boolean_dtypes(),
        integer_dtypes(),
        unsigned_integer_dtypes(),
        floating_dtypes(),
        complex_number_dtypes(),
        datetime64_dtypes(),
        timedelta64_dtypes(),
    )


def defines_dtype_strategy(strat: T) -> T:
    @defines_strategy()
    @proxies(strat)
    def inner(*args, **kwargs):
        return strat(*args, **kwargs).map(np.dtype)

    return inner


@defines_dtype_strategy
def boolean_dtypes() -> st.SearchStrategy[np.dtype]:
    return st.just("?")


def dtype_factory(kind, sizes, valid_sizes, endianness):
    # Utility function, shared logic for most integer and string types
    valid_endian = ("?", "<", "=", ">")
    check_argument(
        endianness in valid_endian,
        "Unknown endianness: was {}, must be in {}",
        endianness,
        valid_endian,
    )
    if valid_sizes is not None:
        if isinstance(sizes, int):
            sizes = (sizes,)
        check_argument(sizes, "Dtype must have at least one possible size.")
        check_argument(
            all(s in valid_sizes for s in sizes),
            "Invalid sizes: was {} must be an item or sequence in {}",
            sizes,
            valid_sizes,
        )
        if all(isinstance(s, int) for s in sizes):
            sizes = sorted({s // 8 for s in sizes})
    strat = st.sampled_from(sizes)
    if "{}" not in kind:
        kind += "{}"
    if endianness == "?":
        return strat.map(("<" + kind).format) | strat.map((">" + kind).format)
    return strat.map((endianness + kind).format)


@defines_dtype_strategy
def unsigned_integer_dtypes(
    *, endianness: str = "?", sizes: Sequence[int] = (8, 16, 32, 64)
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for unsigned integer dtypes.

    endianness may be ``<`` for little-endian, ``>`` for big-endian,
    ``=`` for native byte order, or ``?`` to allow either byte order.
    This argument only applies to dtypes of more than one byte.

    sizes must be a collection of integer sizes in bits.  The default
    (8, 16, 32, 64) covers the full range of sizes.
    """
    return dtype_factory("u", sizes, (8, 16, 32, 64), endianness)


@defines_dtype_strategy
def integer_dtypes(
    *, endianness: str = "?", sizes: Sequence[int] = (8, 16, 32, 64)
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for signed integer dtypes.

    endianness and sizes are treated as for
    :func:`unsigned_integer_dtypes`.
    """
    return dtype_factory("i", sizes, (8, 16, 32, 64), endianness)


@defines_dtype_strategy
def floating_dtypes(
    *, endianness: str = "?", sizes: Sequence[int] = (16, 32, 64)
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for floating-point dtypes.

    sizes is the size in bits of floating-point number.  Some machines support
    96- or 128-bit floats, but these are not generated by default.

    Larger floats (96 and 128 bit real parts) are not supported on all
    platforms and therefore disabled by default.  To generate these dtypes,
    include these values in the sizes argument.
    """
    return dtype_factory("f", sizes, (16, 32, 64, 96, 128), endianness)


@defines_dtype_strategy
def complex_number_dtypes(
    *, endianness: str = "?", sizes: Sequence[int] = (64, 128)
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for complex-number dtypes.

    sizes is the total size in bits of a complex number, which consists
    of two floats.  Complex halves (a 16-bit real part) are not supported
    by numpy and will not be generated by this strategy.
    """
    return dtype_factory("c", sizes, (64, 128, 192, 256), endianness)


@check_function
def validate_time_slice(max_period, min_period):
    check_argument(
        max_period in TIME_RESOLUTIONS,
        "max_period {} must be a valid resolution in {}",
        max_period,
        TIME_RESOLUTIONS,
    )
    check_argument(
        min_period in TIME_RESOLUTIONS,
        "min_period {} must be a valid resolution in {}",
        min_period,
        TIME_RESOLUTIONS,
    )
    start = TIME_RESOLUTIONS.index(max_period)
    end = TIME_RESOLUTIONS.index(min_period) + 1
    check_argument(
        start < end,
        "max_period {} must be earlier in sequence {} than min_period {}",
        max_period,
        TIME_RESOLUTIONS,
        min_period,
    )
    return TIME_RESOLUTIONS[start:end]


@defines_dtype_strategy
def datetime64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for datetime64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "datetime64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def timedelta64_dtypes(
    *, max_period: str = "Y", min_period: str = "ns", endianness: str = "?"
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for timedelta64 dtypes, with various precisions from
    year to attosecond."""
    return dtype_factory(
        "timedelta64[{}]",
        validate_time_slice(max_period, min_period),
        TIME_RESOLUTIONS,
        endianness,
    )


@defines_dtype_strategy
def byte_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for generating bytestring dtypes, of various lengths
    and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("S", list(range(min_len, max_len + 1)), None, endianness)


@defines_dtype_strategy
def unicode_string_dtypes(
    *, endianness: str = "?", min_len: int = 1, max_len: int = 16
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for generating unicode string dtypes, of various
    lengths and byteorder.

    While Hypothesis' string strategies can generate empty strings, string
    dtypes with length 0 indicate that size is still to be determined, so
    the minimum length for string dtypes is 1.
    """
    order_check("len", 1, min_len, max_len)
    return dtype_factory("U", list(range(min_len, max_len + 1)), None, endianness)


def _no_title_is_name_of_a_titled_field(ls):
    seen = set()
    for title_and_name, *_ in ls:
        if isinstance(title_and_name, tuple):
            if seen.intersection(title_and_name):  # pragma: no cover
                # Our per-element filters below make this as rare as possible,
                # so it's not always covered.
                return False
            seen.update(title_and_name)
    return True


@defines_dtype_strategy
def array_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    min_size: int = 1,
    max_size: int = 5,
    allow_subarrays: bool = False,
) -> st.SearchStrategy[np.dtype]:
    """Return a strategy for generating array (compound) dtypes, with members
    drawn from the given subtype strategy."""
    order_check("size", 0, min_size, max_size)
    # The empty string is replaced by f{idx}; see #1963 for details.  Much easier to
    # insist that field names be unique and just boost f{idx} strings manually.
    field_names = st.integers(0, 127).map("f{}".format) | st.text(min_size=1)
    name_titles = st.one_of(
        field_names,
        st.tuples(field_names, field_names).filter(lambda ns: ns[0] != ns[1]),
    )
    elements = st.tuples(name_titles, subtype_strategy)
    if allow_subarrays:
        elements |= st.tuples(
            name_titles, subtype_strategy, array_shapes(max_dims=2, max_side=2)
        )
    return st.lists(
        elements=elements,
        min_size=min_size,
        max_size=max_size,
        unique_by=(
            # Deduplicate by both name and title for efficiency before filtering.
            # (Field names must be unique, as must titles, and no intersections)
            lambda d: d[0] if isinstance(d[0], str) else d[0][0],
            lambda d: d[0] if isinstance(d[0], str) else d[0][1],
        ),
    ).filter(_no_title_is_name_of_a_titled_field)


@defines_strategy()
def nested_dtypes(
    subtype_strategy: st.SearchStrategy[np.dtype] = scalar_dtypes(),
    *,
    max_leaves: int = 10,
    max_itemsize: Optional[int] = None,
) -> st.SearchStrategy[np.dtype]:
    """Return the most-general dtype strategy.

    Elements drawn from this strategy may be simple (from the
    subtype_strategy), or several such values drawn from
    :func:`array_dtypes` with ``allow_subarrays=True``. Subdtypes in an
    array dtype may be nested to any depth, subject to the max_leaves
    argument.
    """
    return st.recursive(
        subtype_strategy,
        lambda x: array_dtypes(x, allow_subarrays=True),
        max_leaves=max_leaves,
    ).filter(lambda d: max_itemsize is None or d.itemsize <= max_itemsize)


valid_tuple_axes = _array_helpers.valid_tuple_axes
valid_tuple_axes.__doc__ = f"""
    Return a strategy for generating permissible tuple-values for the
    ``axis`` argument for a numpy sequential function (e.g.
    :func:`numpy:numpy.sum`), given an array of the specified
    dimensionality.

    {valid_tuple_axes.__doc__}
    """


broadcastable_shapes = _array_helpers.broadcastable_shapes

mutually_broadcastable_shapes = _array_helpers.mutually_broadcastable_shapes
mutually_broadcastable_shapes.__doc__ = f"""
    {mutually_broadcastable_shapes.__doc__}

    **Use with Generalised Universal Function signatures**

    A :np-ref:`universal function <ufuncs.html>` (or ufunc for short) is a function
    that operates on ndarrays in an element-by-element fashion, supporting array
    broadcasting, type casting, and several other standard features.
    A :np-ref:`generalised ufunc <c-api/generalized-ufuncs.html>` operates on
    sub-arrays rather than elements, based on the "signature" of the function.
    Compare e.g. :obj:`numpy:numpy.add` (ufunc) to :obj:`numpy:numpy.matmul` (gufunc).

    To generate shapes for a gufunc, you can pass the ``signature`` argument instead of
    ``num_shapes``.  This must be a gufunc signature string; which you can write by
    hand or access as e.g. ``np.matmul.signature`` on generalised ufuncs.

    In this case, the ``side`` arguments are applied to the 'core dimensions' as well,
    ignoring any frozen dimensions.  ``base_shape``  and the ``dims`` arguments are
    applied to the 'loop dimensions', and if necessary, the dimensionality of each
    shape is silently capped to respect the 32-dimension limit.

    The generated ``result_shape`` is the real result shape of applying the gufunc
    to arrays of the generated ``input_shapes``, even where this is different to
    broadcasting the loop dimensions.

    gufunc-compatible shapes shrink their loop dimensions as above, towards omitting
    optional core dimensions, and smaller-size core dimensions.

    .. code-block:: pycon

        >>> # np.matmul.signature == "(m?,n),(n,p?)->(m?,p?)"
        >>> for _ in range(3):
        ...     mutually_broadcastable_shapes(signature=np.matmul.signature).example()
        BroadcastableShapes(input_shapes=((2,), (2,)), result_shape=())
        BroadcastableShapes(input_shapes=((3, 4, 2), (1, 2)), result_shape=(3, 4))
        BroadcastableShapes(input_shapes=((4, 2), (1, 2, 3)), result_shape=(4, 3))

    """

basic_indices = _array_helpers.make_basic_indices(allow_0d_index=True)
basic_indices.__doc__ = f"""
    Return a strategy for :np-ref:`basic indexes <arrays.indexing.html>` of
    arrays with the specified shape, which may include dimensions of size zero.

    {basic_indices.__doc__}
    """


@defines_strategy()
def integer_array_indices(
    shape: Shape,
    *,
    result_shape: st.SearchStrategy[Shape] = array_shapes(),
    dtype: np.dtype = "int",
) -> st.SearchStrategy[Tuple[np.ndarray, ...]]:
    """Return a search strategy for tuples of integer-arrays that, when used
    to index into an array of shape ``shape``, given an array whose shape
    was drawn from ``result_shape``.

    Examples from this strategy shrink towards the tuple of index-arrays::

        len(shape) * (np.zeros(drawn_result_shape, dtype), )

    * ``shape`` a tuple of integers that indicates the shape of the array,
      whose indices are being generated.
    * ``result_shape`` a strategy for generating tuples of integers, which
      describe the shape of the resulting index arrays. The default is
      :func:`~hypothesis.extra.numpy.array_shapes`.  The shape drawn from
      this strategy determines the shape of the array that will be produced
      when the corresponding example from ``integer_array_indices`` is used
      as an index.
    * ``dtype`` the integer data type of the generated index-arrays. Negative
      integer indices can be generated if a signed integer type is specified.

    Recall that an array can be indexed using a tuple of integer-arrays to
    access its members in an arbitrary order, producing an array with an
    arbitrary shape. For example:

    .. code-block:: pycon

        >>> from numpy import array
        >>> x = array([-0, -1, -2, -3, -4])
        >>> ind = (array([[4, 0], [0, 1]]),)  # a tuple containing a 2D integer-array
        >>> x[ind]  # the resulting array is commensurate with the indexing array(s)
        array([[-4,  0],
               [0, -1]])

    Note that this strategy does not accommodate all variations of so-called
    'advanced indexing', as prescribed by NumPy's nomenclature.  Combinations
    of basic and advanced indexes are too complex to usefully define in a
    standard strategy; we leave application-specific strategies to the user.
    Advanced-boolean indexing can be defined as ``arrays(shape=..., dtype=bool)``,
    and is similarly left to the user.
    """
    check_type(tuple, shape, "shape")
    check_argument(
        shape and all(isinstance(x, int) and x > 0 for x in shape),
        f"shape={shape!r} must be a non-empty tuple of integers > 0",
    )
    check_strategy(result_shape, "result_shape")
    check_argument(
        np.issubdtype(dtype, np.integer), f"dtype={dtype!r} must be an integer dtype"
    )
    signed = np.issubdtype(dtype, np.signedinteger)

    def array_for(index_shape, size):
        return arrays(
            dtype=dtype,
            shape=index_shape,
            elements=st.integers(-size if signed else 0, size - 1),
        )

    return result_shape.flatmap(
        lambda index_shape: st.tuples(*(array_for(index_shape, size) for size in shape))
    )
