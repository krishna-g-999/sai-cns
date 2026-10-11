"""The project's activity label rule, in one place.

Every endpoint's class labels come from this rule, so it decides what the panel is trained on. It
used to exist as six near-copies, one per fetcher, each re-declaring the two cuts. They agreed on
the cuts, but they did not all agree on the rule: fetch_natural_products matched only a bare ">"
and "<", so a ">=" bound fell through to the exact-value branch and was read as a potency. That is
precisely the defect the rule exists to prevent, and it is the one that once lost 253 measured
non-binders for AChE alone. No ">=" or "<=" appears in the data fetched so far, so no label was
ever wrong because of it, but ChEMBL's standard_relation does emit both, so a re-fetch could have
triggered it silently.

The rule
--------
Potencies are on the pChEMBL scale, -log10(molar), so a larger value is more potent.

  - an exact value is active at or above ACTIVE_CUT and inactive at or below INACTIVE_CUT; the band
    between the two cuts is ambiguous and is discarded rather than guessed
  - a censored bound settles a class only when the whole interval it defines falls on one side of
    the relevant cut. "IC50 > 10 uM" is a bound at pChEMBL 5.0: everything below it, so the
    compound is unambiguously inactive. "IC50 > 100 nM" is a bound at pChEMBL 7.0, and the true
    value could be either class, so it is undecidable and discarded
  - the relation symbol refers to the original concentration, so it inverts on the pChEMBL scale:
    a ">" on IC50 is a "<" on potency. That inversion is why passing a bound to the exact-value
    branch is a silent mislabelling rather than an obvious error

Any relation that is not a recognised bound is treated as exact. That covers "=", "~", the empty
string, "nan", "n.a." and None, which is what every previous copy did.
"""
from __future__ import annotations

ACTIVE_CUT: float = 6.0
INACTIVE_CUT: float = 5.0

# The relation symbols that denote a censored bound, as they appear after normalisation. The
# original concentration's ">" becomes a bound *below* which the potency lies, and vice versa.
WEAKER_THAN = (">", ">=")   # potency is below the quoted bound  -> can only settle "inactive"
STRONGER_THAN = ("<", "<=")  # potency is above the quoted bound  -> can only settle "active"


def label_from(p: float, relation: str | None) -> int | None:
    """Return 1 (active), 0 (inactive), or None when the measurement cannot settle a class.

    ``p`` is a pChEMBL-scale potency, or the bound itself for a censored record. ``relation`` is
    the symbol attached to the original concentration; it is normalised here, so callers may pass
    None, an empty string or a padded symbol.
    """
    rel = (relation or "=").strip()
    if rel in WEAKER_THAN:
        return 0 if p <= INACTIVE_CUT else None
    if rel in STRONGER_THAN:
        return 1 if p >= ACTIVE_CUT else None
    if p >= ACTIVE_CUT:
        return 1
    if p <= INACTIVE_CUT:
        return 0
    return None


def bound_settles_inactive(bound: float) -> bool:
    """Whether a ">" bound places the whole interval below the inactive cut.

    The same rule as ``label_from(bound, ">") == 0``, named for the two fetchers that need only
    this half of it and previously inlined it as a bare comparison against a locally declared cut.

    One deliberate behaviour change from that inlined form. The fetchers wrote
    ``if bound > INACTIVE_CUT: discard``, which is this function negated for every real number but
    not for NaN: ``nan > 5.0`` is False, so a NaN bound was kept and written out as a confident
    inactive label, whereas ``not (nan <= 5.0)`` is True, so it is now discarded as undecidable.

    A NaN bound reaches those call sites only if ChEMBL returns a non-numeric standard_value that
    still parses through float(), which has not been observed in any fetched table. The new
    behaviour is the one the rule implies: a measurement that is not a number cannot settle a
    class, and it is what label_from already does, since NaN fails both of its comparisons and
    falls through to None. The two halves of the rule now agree on that.
    """
    return bound <= INACTIVE_CUT
