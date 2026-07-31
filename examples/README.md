# Examples

```bash
cd python && PYTHONPATH=src python3 ../examples/demo.py
```

`demo.py` runs three stages: a null market (the system must find nothing), a
realistic edge (the system finds it but cannot prove it, and so stands aside),
and a recovery test at increasing edge strength (the gate opens once the edge is
demonstrable).

Equivalent CLI entry points:

```bash
PYTHONPATH=src python3 -m tia.cli demo             # the same walkthrough
PYTHONPATH=src python3 -m tia.cli config           # parameter manifest + hash
PYTHONPATH=src python3 -m tia.cli run   --csv data.csv
PYTHONPATH=src python3 -m tia.cli train --csv data.csv
PYTHONPATH=src python3 -m tia.cli validate --csv data.csv --folds 4
PYTHONPATH=src python3 -m tia.cli export           # writes pine/frozen_model.json
```

CSV columns, case-insensitive and in any order: a time column named one of
`timestamp / time / date / datetime / open_time`, plus `open`, `high`, `low`,
`close`, `volume`. Rows that fail OHLC consistency are skipped, not repaired —
repairing means guessing, and a guess that reaches the feature kernel is
indistinguishable from a signal.

**Expect `NO TRADE`.** On a few thousand bars of one instrument, that is the
correct output, and `docs/01-THEORY.md` §12 explains why a single-instrument
run cannot tell you whether the edge is real.
