"""Train the orexin OX1 and OX2 binder models.

This was a verbatim copy of train_binders_hybrid.py differing only in its target list, which meant
every correction to the training procedure had to be made in three places or silently applied to two
thirds of the panel. It is kept as an entry point because the run that produced the deployed orexin models was invoked through it, but it now delegates rather than
duplicating: the parent already accepts targets on the command line.

Equivalent to:  python src/sai_cns/models/train_binders_hybrid.py OX1 OX2
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_binders_hybrid  # noqa: E402

TARGETS = ['OX1', 'OX2']

if __name__ == "__main__":
    train_binders_hybrid.TARGETS = TARGETS
    sys.argv = [sys.argv[0]] + TARGETS
    train_binders_hybrid.main()
