"""Comparing against the reference in a way that survives a change of compiler.

A pure relative tolerance is the wrong test for these kernels. gcc and clang evaluate
log, exp and division to results that differ in the last bits, and several quantities
here are differences of nearly equal numbers -- the absorption optical depth
tau*(1 - ssa) when ssa is close to one, the interpolation weight (1 - feta) when feta
is close to one, the adding recurrences. A few ulp of input difference becomes 1e-8
relative on the result, while the absolute difference stays at 1e-15.

Comparing to a tolerance on the *field's own scale* is both stricter where it matters
and portable: it says the two implementations agree to one part in 1e12 of the largest
value present, which is the physically meaningful statement, rather than demanding that
a value 1e11 times smaller than its neighbours also match to 1e-12.

Cases that compare fluxes against a stored reference keep their own absolute tolerance
in W/m2, which is already scale-based.
"""
import numpy as np

# One part in 1e12 of the field's scale. Roughly 4500 times double epsilon, so it still
# fails on anything but last-bit differences.
DEFAULT_RTOL = 1e-12

# The solvers walk a sequential recurrence down the column -- the adding method, and
# the longwave transport sweeps -- which amplifies whatever difference the inputs
# carry. Building the same source with gcc and with clang gives fluxes differing by
# 1.9e-11 relative on a 12-layer column with randomised optical properties, because
# their log and exp differ in the last bits. That is the quantity this tolerance has to
# clear; the margin below is about fifty times it.
#
# It is still ten million times tighter than any physically meaningful flux difference,
# and a genuine error in a solver shows up as an O(1) discrepancy, not a 1e-10 one.
RECURRENCE_RTOL = 1e-9


def assert_close(actual, expected, rtol=DEFAULT_RTOL, err_msg=''):
    """Assert agreement to rtol, relative to each element and to the field's scale."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    scale = np.abs(expected).max() if expected.size else 0.0

    np.testing.assert_allclose(
        actual, expected, rtol=rtol, atol=rtol*scale, err_msg=err_msg)
