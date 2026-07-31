"""A guided walkthrough of TI-A.

Run it:

    cd python && PYTHONPATH=src python3 ../examples/demo.py

Same content as ``python3 -m tia.cli demo``, laid out for reading rather than
for the terminal. The three stages are the scientific argument in miniature:

1.  **Null market.** A martingale with realistic volatility clustering, fat
    tails and an intraday volume profile, but a conditional mean of exactly
    zero. The system must find nothing. A framework that trades here has a
    look-ahead bug or a cost-model artifact, and this is the cheapest test that
    catches either.

2.  **A realistic edge.** The generator alternates between persistence and
    reversion at a strength comparable to information coefficients that are
    actually achievable. Net expectancy comes out positive -- and the system
    still emits nothing, because at ~50 effective observations the t-statistic
    is around 1 and the *lower credible bound* is negative.

3.  **Recovery test.** The same generator at increasing edge strength. The gate
    opens once the edge is demonstrable. Without this stage, a system that never
    trades is indistinguishable from one whose gate is stuck shut.

Stage 2 is the one worth sitting with. The system is not refusing because the
edge is absent; it is refusing because the edge cannot be *demonstrated*. That
distinction is the whole design.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "src"))

from tia.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main(["demo", "--bars", "2500"]))
