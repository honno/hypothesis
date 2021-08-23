RELEASE_TYPE: minor

This minor release introduces strategies for array/tensor libraries adopting the
`Array API <https://data-apis.org/>`_ standard, closing :issue:`3037`. They are
available in the `array_api` extra, working much like the existing
:doc:`strategies for NumPy <numpy>` except requiring users to specify or pass the
implementing module.
