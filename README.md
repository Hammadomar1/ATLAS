# ATLAS

Prototype implementation of ATLAS Part 1: a two-time-base reflex controller with non-blocking onboard observer and offboard advisory modulation.

## Structure

- `atlas/`: core modules (reflex engine, supervisor, safety shield, parameter server, observer, offboard meta).
- `atlas/sim.py`: runnable simulation harness.
- `tests/`: unit tests for critical invariants (projection, shield, delay/dropout).

## Quick start

```bash
python -m atlas.sim
```

## Tests

```bash
python -m unittest
```
