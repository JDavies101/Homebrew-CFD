# Introduction

Homebrew CFD is a GPU lattice Boltzmann solver for external aerodynamics: wings, bodies and wheels in free
stream or in ground effect. It runs D3Q19 on CUDA, with Bouzidi interpolated walls, WALE large-eddy
simulation and a log-law wall model.

## What to expect from the numbers

The solver is validated against analytic solutions and published benchmarks (DESIGN.md, section 7). It is
built for relative comparisons and credible trends: one ride height against another, one wing angle against
another. Absolute drag and downforce are not certification grade.

The numbers are tied to the grid. Every summary shows the resolved Reynolds number next to the physical one:
when the grid cannot carry the physical Reynolds number, the case runs at the highest Reynolds number the grid
resolves, and says so.

## Offline

Homebrew CFD makes no network calls and has no accounts. Cases, geometry and results stay on this machine,
in Documents\Homebrew CFD Projects.

## This manual

1. Introduction (this page)
2. Getting started: the start page, projects, the main window
3. Tutorial: wing in ground effect
4. Tutorial: importing an STL
5. Reference: every setting, the run folder, validated cases, known limits

The manual covers version 0.3.
