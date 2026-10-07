# FRE-GY-9743-Assignments-2026fall
Assignments for NYU, FRE-GY 9743, Fall 2026.


## Assignment 3 — implementation notes

The six `#TODO` blocks in `fixedincomelib/product/` (the two cashflow products
and the display visitor) are implemented. Documentation lives in [`docs/`](docs/):

- [docs/hw3-implementation.md](docs/hw3-implementation.md) — what each method
  does, why, and how it was verified.
- [docs/hw1-hw2-code-reuse.md](docs/hw1-hw2-code-reuse.md) — the one file copied
  from HW1, and what was already carried over from HW2.
- [docs/repo-structure.md](docs/repo-structure.md) — import hazards that persist
  in this package layout, plus the verified environment.

Run the regression tests with:

```bash
python tests/test_products.py
```
