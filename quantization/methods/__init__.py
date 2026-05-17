"""Per-method LightCompress yml builders.

Importing this package registers every method module in
:mod:`quantization.registry` via :func:`register` calls in each module.
"""
from quantization.methods import (  # noqa: F401  (side-effect imports)
    adadim,
    awq,
    dgq,
    gptq,
    hqq,
    llmint8,
    normtweaking,
    omniquant,
    osplus,
    quarot,
    quik,
    rtn,
    smoothquant,
    spqr,
    tesseraq,
)
