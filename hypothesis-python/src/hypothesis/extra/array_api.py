import math
from collections import defaultdict
from functools import update_wrapper, wraps
from numbers import Real
from types import SimpleNamespace
from typing import (
    Any,
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

from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.errors import HypothesisWarning, InvalidArgument
from hypothesis.internal.coverage import check_function
from hypothesis.internal.conjecture import utils as cu
from hypothesis.internal.validation import check_type, check_valid_interval
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


Shape = Tuple[int, ...]
BasicIndex = Tuple[Union[int, slice, None, "ellipsis"], ...]  # noqa: F821


class BroadcastableShapes(NamedTuple):
    input_shapes: Tuple[Shape, ...]
    result_shape: Shape


INT_NAMES = ["int8", "int16", "int32", "int64"]
UINT_NAMES = ["uint8", "uint16", "uint32", "uint64"]
ALL_INT_NAMES = INT_NAMES + UINT_NAMES
FLOAT_NAMES = ["float32", "float64"]
NUMERIC_NAMES = ALL_INT_NAMES + FLOAT_NAMES
DTYPE_NAMES = ["bool"] + NUMERIC_NAMES


@check_function
def infer_xp_is_compliant(xp):
    try:
        array = xp.zeros(1)
        array.__array_namespace__()
    except Exception:
        warn(
            f"Could not determine whether module {xp} is an Array API library",
            HypothesisWarning,
        )


@check_function
def check_xp_attributes(xp, attributes: List[str]):
    missing_attrs = [attr for attr in attributes if not hasattr(xp, attr)]
    if len(missing_attrs) > 0:
        f_attrs = ", ".join(missing_attrs)
        raise InvalidArgument(
            f"Array module {xp} does not have required attributes: {f_attrs}"
        )


def partition_attributes_and_stubs(
    xp, attributes: Iterable[str]
) -> Tuple[List[Any], List[str]]:
    non_stubs = []
    stubs = []
    for attr in attributes:
        try:
            non_stubs.append(getattr(xp, attr))
        except AttributeError:
            stubs.append(attr)

    return non_stubs, stubs


@check_function
def warn_on_missing_dtypes(xp, stubs: List[str]):
    f_stubs = ", ".join(stubs)
    warn(
        f"Array module {xp} does not have the following "
        f"dtypes in its namespace: {f_stubs}",
        HypothesisWarning,
    )


def find_castable_builtin_for_dtype(xp, dtype: Type) -> Type[Union[bool, int, float]]:
    """Returns builtin type which can have values that are castable to the given
    dtype, according to :xp-ref:`type promotion rules <type_promotion.html>`.

    ``float`` is always returned for floating dtypes, as opposed to ``int``.
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


def dtype_from_name(xp, name: str) -> Type:
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


def pretty_xp_repr(func):
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
    xp,
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

    def check_min_value(info_obj):
        assert isinstance(min_value, Real)
        if min_value < info_obj.min:
            raise InvalidArgument(
                f"dtype {dtype} requires min_value={min_value} "
                f"to be at least {info_obj.min}"
            )

    def check_max_value(info_obj):
        assert isinstance(max_value, Real)
        if max_value > info_obj.max:
            raise InvalidArgument(
                f"dtype {dtype} requires max_value={max_value} "
                f"to be at most {info_obj.max}"
            )

    if builtin is bool:
        return st.booleans()
    elif builtin is int:
        iinfo = xp.iinfo(dtype)
        kw = {}
        if min_value is None:
            kw["min_value"] = iinfo.min
        else:
            check_min_value(iinfo)
            kw["min_value"] = min_value
        if max_value is None:
            kw["max_value"] = iinfo.max
        else:
            check_max_value(iinfo)
            kw["max_value"] = max_value
        return st.integers(**kw)
    else:
        finfo = xp.finfo(dtype)
        kw = {}
        # Whilst we know the boundary values of float dtypes we do not assign
        # them to the floats() strategy by default - passing min/max values will
        # modify test case reduction behaviour so that simple bugs may become
        # harder for users to identify.
        if min_value is not None:
            check_min_value(finfo)
            kw["min_value"] = min_value
        if max_value is not None:
            check_max_value(finfo)
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
    xp,
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
        assert isinstance(elements, st.SearchStrategy)
        if unique or not elements.has_reusable_values:
            fill = st.nothing()
        else:
            fill = elements
    check_strategy(fill, "fill")

    return ArrayStrategy(xp, elements, dtype, shape, fill, unique)


@check_function
def order_check(name, floor, min_, max_):
    if floor > min_:
        raise InvalidArgument(f"min_{name} must be at least {floor} but was {min_}")
    if min_ > max_:
        raise InvalidArgument(f"min_{name}={min_} is larger than max_{name}={max_}")


@defines_strategy()
def array_shapes(
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for array shapes (tuples of int >= 1).

    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``min_dims + 2``.
    * ``min_side`` is the smallest size that a dimension can possess.
    * ``max_side`` is the largest size that a dimension can possess,
      defaulting to ``min_side + 5``.
    """
    check_type(int, min_dims, "min_dims")
    check_type(int, min_side, "min_side")

    if max_dims is None:
        max_dims = min_dims + 2
    check_type(int, max_dims, "max_dims")

    if max_side is None:
        max_side = min_side + 5
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    return st.lists(
        st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims
    ).map(tuple)


@check_function
def check_dtypes(xp, dtypes: List[Type], stubs: List[str]):
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
def scalar_dtypes(xp) -> st.SearchStrategy[Type]:
    """Return a strategy for all :xp-ref:`valid dtype <data_types.html>` objects."""
    infer_xp_is_compliant(xp)
    return st.one_of(boolean_dtypes(xp), numeric_dtypes(xp))


@pretty_xp_repr
@defines_strategy()
def boolean_dtypes(xp) -> st.SearchStrategy[Type]:
    infer_xp_is_compliant(xp)
    try:
        return st.just(xp.bool)
    except AttributeError:
        raise InvalidArgument(
            f"Array module {xp} does not have a bool dtype in its namespace"
        ) from None


@pretty_xp_repr
@defines_strategy()
def numeric_dtypes(xp) -> st.SearchStrategy[Type]:
    """Return a strategy for all numeric dtype objects."""
    infer_xp_is_compliant(xp)
    return st.one_of(
        integer_dtypes(xp),
        unsigned_integer_dtypes(xp),
        floating_dtypes(xp),
    )


@check_function
def check_valid_sizes(category: str, sizes: Sequence[int], valid_sizes: Sequence[int]):
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
    xp, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
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
    xp, *, sizes: Union[int, Sequence[int]] = (8, 16, 32, 64)
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
    xp, *, sizes: Union[int, Sequence[int]] = (32, 64)
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


@defines_strategy()
def valid_tuple_axes(
    ndim: int,
    *,
    min_size: int = 0,
    max_size: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for permissible tuple-values for the ``axis``
    argument in Array API sequential methods e.g. ``sum``, given the specified
    dimensionality.

    All tuples will have a length >= ``min_size`` and <= ``max_size``. The default
    value for ``max_size`` is ``ndim``.

    Examples from this strategy shrink towards an empty tuple, which render most
    sequential functions as no-ops.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

      >>> [valid_tuple_axes(3).example() for i in range(4)]
      [(-3, 1), (0, 1, -1), (0, 2), (0, -2, 2)]

    ``valid_tuple_axes`` can be joined with other strategies to generate
    any type of valid axis object, i.e. integers, tuples, and ``None``:

    .. code-block:: python

      any_axis_strategy = none() | integers(-ndim, ndim - 1) | valid_tuple_axes(ndim)

    """
    if max_size is None:
        max_size = ndim
    check_type(int, ndim, "ndim")
    check_type(int, min_size, "min_size")
    check_type(int, max_size, "max_size")
    order_check("size", 0, min_size, max_size)
    check_valid_interval(max_size, ndim, "max_size", "ndim")
    axes = st.integers(0, max(0, 2 * ndim - 1)).map(
        lambda x: x if x < ndim else x - 2 * ndim
    )
    return st.lists(
        axes, min_size=min_size, max_size=max_size, unique_by=lambda x: x % ndim
    ).map(tuple)


class MutuallyBroadcastableShapesStrategy(st.SearchStrategy):
    def __init__(
        self,
        num_shapes,
        base_shape=(),
        min_dims=0,
        max_dims=None,
        min_side=1,
        max_side=None,
    ):
        self.base_shape = base_shape
        self.num_shapes = num_shapes
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.min_side = min_side
        self.max_side = max_side

        self.side_strat = st.integers(min_side, max_side)
        self.size_one_allowed = self.min_side <= 1 <= self.max_side

    def do_draw(self, data):
        # All shapes are handled in column-major order; i.e. they are reversed
        base_shape = self.base_shape[::-1]
        result_shape = list(base_shape)
        shapes = [[] for _ in range(self.num_shapes)]
        use = [True for _ in range(self.num_shapes)]

        for dim_count in range(1, self.max_dims + 1):
            dim = dim_count - 1

            # We begin by drawing a valid dimension-size for the given
            # dimension. This restricts the variability across the shapes
            # at this dimension such that they can only choose between
            # this size and a singleton dimension.
            if len(base_shape) < dim_count or base_shape[dim] == 1:
                # dim is unrestricted by the base-shape: shrink to min_side
                dim_side = data.draw(self.side_strat)
            elif base_shape[dim] <= self.max_side:
                # dim is aligned with non-singleton base-dim
                dim_side = base_shape[dim]
            else:
                # only a singleton is valid in alignment with the base-dim
                dim_side = 1

            for shape_id, shape in enumerate(shapes):
                # Populating this dimension-size for each shape, either
                # the drawn size is used or, if permitted, a singleton
                # dimension.
                if dim_count <= len(base_shape) and self.size_one_allowed:
                    # aligned: shrink towards size 1
                    side = data.draw(st.sampled_from([1, dim_side]))
                else:
                    side = dim_side

                # Use a trick where where a biased coin is queried to see
                # if the given shape-tuple will continue to be grown. All
                # of the relevant draws will still be made for the given
                # shape-tuple even if it is no longer being added to.
                # This helps to ensure more stable shrinking behavior.
                if self.min_dims < dim_count:
                    use[shape_id] &= cu.biased_coin(
                        data, 1 - 1 / (1 + self.max_dims - dim)
                    )

                if use[shape_id]:
                    shape.append(side)
                    if len(result_shape) < len(shape):
                        result_shape.append(shape[-1])
                    elif shape[-1] != 1 and result_shape[dim] == 1:
                        result_shape[dim] = shape[-1]
            if not any(use):
                break

        result_shape = result_shape[: max(map(len, [self.base_shape] + shapes))]

        assert len(shapes) == self.num_shapes
        assert all(self.min_dims <= len(s) <= self.max_dims for s in shapes)
        assert all(self.min_side <= s <= self.max_side for side in shapes for s in side)

        return BroadcastableShapes(
            input_shapes=tuple(tuple(reversed(shape)) for shape in shapes),
            result_shape=tuple(reversed(result_shape)),
        )


@defines_strategy()
def broadcastable_shapes(
    shape: Shape,
    *,
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[Shape]:
    """Return a strategy for shapes that are broadcast-compatible with the
    provided shape.

    Examples from this strategy shrink towards a shape with length ``min_dims``.
    The size of an aligned dimension shrinks towards size ``1``. The size of an
    unaligned dimension shrink towards ``min_side``.

    * ``shape`` is a tuple of integers.
    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``min(32, max(len(shape), min_dims) + 2)``.
    * ``min_side`` is the smallest size that an unaligned dimension can possess.
    * ``max_side`` is the largest size that an unaligned dimension can possess,
      defaulting to 2 plus the size of the largest aligned dimension.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> [broadcastable_shapes(shape=(2, 3)).example() for i in range(5)]
        [(1, 3), (), (2, 3), (2, 1), (4, 1, 3), (3, )]
    """
    check_type(tuple, shape, "shape")
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")

    strict_check = max_side is None or max_dims is None

    if max_dims is None:
        max_dims = min(32, max(len(shape), min_dims) + 2)
    check_type(int, max_dims, "max_dims")

    if max_side is None:
        max_side = max(shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if strict_check:
        dims = max_dims
        bound_name = "max_dims"
    else:
        dims = min_dims
        bound_name = "min_dims"

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            f"Given shape={shape}, there are no broadcast-compatible "
            f"shapes that satisfy: {bound_name}={dims} and min_side={min_side}"
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in shape[::-1][:dims])
    ):
        raise InvalidArgument(
            f"Given base_shape={shape}, there are no broadcast-compatible "
            f"shapes that satisfy all of {bound_name}={dims}, "
            f"min_side={min_side}, and max_side={max_side}"
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=1,
        base_shape=shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    ).map(lambda x: x.input_shapes[0])


@defines_strategy()
def mutually_broadcastable_shapes(
    num_shapes: int,
    *,
    base_shape: Shape = (),
    min_dims: int = 0,
    max_dims: Optional[int] = None,
    min_side: int = 1,
    max_side: Optional[int] = None,
) -> st.SearchStrategy[BroadcastableShapes]:
    """Return a strategy for a specified number of shapes N that are
    mutually-broadcastable with one another and with the provided base shape.

    * ``num_shapes`` is the number of mutually broadcast-compatible shapes to generate.
    * ``base_shape`` is the shape against which all generated shapes can broadcast.
      The default shape is empty, which corresponds to a scalar and thus does
      not constrain broadcasting at all.
    * ``shape`` is a tuple of integers.
    * ``min_dims`` is the smallest length that the generated shape can possess.
    * ``max_dims`` is the largest length that the generated shape can possess,
      defaulting to ``min(32, max(len(shape), min_dims) + 2)``.
    * ``min_side`` is the smallest size that an unaligned dimension can possess.
    * ``max_side`` is the largest size that an unaligned dimension can possess,
      defaulting to 2 plus the size of the largest aligned dimension.

    The strategy will generate a :obj:`python:typing.NamedTuple` containing:

    * ``input_shapes`` as a tuple of the N generated shapes.
    * ``result_shape`` as the resulting shape produced by broadcasting the N shapes
      with the base shape.

    The following are some examples drawn from this strategy.

    .. code-block:: pycon

        >>> # Draw three shapes where each shape is broadcast-compatible with (2, 3)
        ... strat = mutually_broadcastable_shapes(num_shapes=3, base_shape=(2, 3))
        >>> for _ in range(5):
        ...     print(strat.example())
        BroadcastableShapes(input_shapes=((4, 1, 3), (4, 2, 3), ()), result_shape=(4, 2, 3))
        BroadcastableShapes(input_shapes=((3,), (1, 3), (2, 3)), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((), (), ()), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((3,), (), (3,)), result_shape=(2, 3))
        BroadcastableShapes(input_shapes=((1, 2, 3), (3,), ()), result_shape=(1, 2, 3))

    """

    check_type(int, num_shapes, "num_shapes")
    if num_shapes < 1:
        raise InvalidArgument(f"num_shapes={num_shapes} must be at least 1")

    check_type(tuple, base_shape, "base_shape")
    check_type(int, min_side, "min_side")
    check_type(int, min_dims, "min_dims")

    strict_check = max_dims is not None

    if max_dims is None:
        max_dims = min(32, max(len(base_shape), min_dims) + 2)
    check_type(int, max_dims, "max_dims")

    if max_side is None:
        max_side = max(base_shape[-max_dims:] + (min_side,)) + 2
    check_type(int, max_side, "max_side")

    order_check("dims", 0, min_dims, max_dims)
    order_check("side", 0, min_side, max_side)

    if strict_check:
        dims = max_dims
        bound_name = "max_dims"
    else:
        dims = min_dims
        bound_name = "min_dims"

    # check for unsatisfiable min_side
    if not all(min_side <= s for s in base_shape[::-1][:dims] if s != 1):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible "
            f"shapes that satisfy: {bound_name}={dims} and min_side={min_side}"
        )

    # check for unsatisfiable [min_side, max_side]
    if not (
        min_side <= 1 <= max_side or all(s <= max_side for s in base_shape[::-1][:dims])
    ):
        raise InvalidArgument(
            f"Given base_shape={base_shape}, there are no broadcast-compatible "
            f"shapes that satisfy all of {bound_name}={dims}, "
            f"min_side={min_side}, and max_side={max_side}"
        )

    if not strict_check:
        # reduce max_dims to exclude unsatisfiable dimensions
        for n, s in zip(range(max_dims), base_shape[::-1]):
            if s < min_side and s != 1:
                max_dims = n
                break
            elif not (min_side <= 1 <= max_side or s <= max_side):
                max_dims = n
                break

    return MutuallyBroadcastableShapesStrategy(
        num_shapes=num_shapes,
        base_shape=base_shape,
        min_dims=min_dims,
        max_dims=max_dims,
        min_side=min_side,
        max_side=max_side,
    )


class IndexStrategy(st.SearchStrategy):
    def __init__(self, shape, min_dims, max_dims, allow_ellipsis, allow_none):
        self.shape = shape
        self.min_dims = min_dims
        self.max_dims = max_dims
        self.allow_ellipsis = allow_ellipsis
        self.allow_none = allow_none

    def do_draw(self, data):
        # General plan: determine the actual selection up front with a straightforward
        # approach that shrinks well, then complicate it by inserting other things.
        result = []
        for dim_size in self.shape:
            if dim_size == 0:
                result.append(slice(None))
                continue
            strategy = st.integers(-dim_size, dim_size - 1) | st.slices(dim_size)
            result.append(data.draw(strategy))
        # Insert some number of new size-one dimensions if allowed
        result_dims = sum(isinstance(idx, slice) for idx in result)
        while (
            self.allow_none
            and result_dims < self.max_dims
            and (result_dims < self.min_dims or data.draw(st.booleans()))
        ):
            i = data.draw(st.integers(0, len(result)))
            result.insert(i, None)
            result_dims += 1
        # Check that we'll have the right number of dimensions; reject if not.
        # It's easy to do this by construction if you don't care about shrinking,
        # which is really important for array shapes.  So we filter instead.
        assume(self.min_dims <= result_dims <= self.max_dims)
        # This is a quick-and-dirty way to insert ..., xor shorten the indexer,
        # but it means we don't have to do any structural analysis.
        if self.allow_ellipsis and data.draw(st.booleans()):
            # Choose an index; then replace all adjacent whole-dimension slices.
            i = j = data.draw(st.integers(0, len(result)))
            while i > 0 and result[i - 1] == slice(None):
                i -= 1
            while j < len(result) and result[j] == slice(None):
                j += 1
            result[i:j] = [Ellipsis]
        else:
            while result[-1:] == [slice(None, None)] and data.draw(st.integers(0, 7)):
                result.pop()
        if len(result) == 1 and data.draw(st.booleans()):
            # Sometimes generate bare element equivalent to a length-one tuple
            return result[0]
        return tuple(result)


@defines_strategy()
def indices(
    shape: Shape,
    *,
    min_dims: int = 1,
    max_dims: Optional[int] = None,
    allow_ellipsis: bool = True,
    allow_none: bool = False,
) -> st.SearchStrategy[BasicIndex]:
    """Return a strategy for :xp-ref:`valid indices <indexing.html>` of
    arrays with the specified shape.

    It generates tuples containing some mix of integers, :obj:`python:slice`
    objects, ``...`` (an ``Ellipsis``), and ``None``. When a length-one tuple
    would be generated, this strategy may instead return the element which will
    index the first axis, e.g. ``5`` instead of ``(5,)``.

    * ``shape`` is the shape of the array that will be indexed, as a tuple of
      integers >= 0. This must be at least two-dimensional for a tuple to be a
      valid index; for one-dimensional arrays use
      :func:`~hypothesis.strategies.slices` instead.
    * ``min_dims`` is the minimum dimensionality of the resulting array from use of
      the generated index.  If ``min_dims == 0``, zero-dimensional arrays are allowed.
    * ``max_dims`` is the the maximum dimensionality of the resulting array,
      defaulting to ``max(len(shape), min_dims) + 2``.
    * ``allow_ellipsis`` specifies whether ``...`` is allowed in the index.
    * ``allow_none`` specifies whether ``None`` is allowed in the index.
    """
    check_type(tuple, shape, "shape")
    if len(shape) == 0:
        raise InvalidArgument("No valid indices for zero-dimensional arrays")
    check_type(bool, allow_ellipsis, "allow_ellipsis")
    check_type(bool, allow_none, "allow_none")
    check_type(int, min_dims, "min_dims")
    if max_dims is None:
        max_dims = min(max(len(shape), min_dims) + 2, 32)
    check_type(int, max_dims, "max_dims")
    order_check("dims", 1, min_dims, max_dims)
    if not all(isinstance(x, int) and x >= 0 for x in shape):
        raise InvalidArgument(
            f"shape={shape!r}, but all dimensions must be of integer size >= 0"
        )
    return IndexStrategy(
        shape,
        min_dims=min_dims,
        max_dims=max_dims,
        allow_ellipsis=allow_ellipsis,
        allow_none=allow_none,
    )


@pretty_xp_repr
def get_strategies_namespace(xp) -> SimpleNamespace:
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
