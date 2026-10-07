# Reference

Lengths are in cells, velocities and times in lattice units (one step, one cell), unless marked.

## Study

Each study has a name, a type (Steady: averaged forces), and its own Solver and Timing settings below. New
studies copy them from the case. A study is stored in the project's `studies\<name>\` folder: `study.json`
and the case file of every run, as run.

| Parametric sweep | Meaning |
|---|---|
| part | the part the parameter belongs to |
| parameter | position y (cells: naca lowest point, stl offset, sphere and cylinder center) or angle (degrees: naca incidence, ahmed slant) |
| values | a list, or start, end and count |

## Solver

| Setting | Meaning |
|---|---|
| collision | regularized (default), trt or bgk |
| inlet | neem_open (default), neem or none |
| start | rest_ramp (default: flow ramped up from rest) or uniform (flow at U everywhere) |
| allow below floor | run with tau under 0.501; only stable with the LES model on |

## Flow

| Setting | Meaning |
|---|---|
| free stream velocity | U; Mach = U √3, keep it near 0.1 or below |
| reynolds number | Re on the reference length; with U and L it sets the viscosity and tau |
| reference length | L, cells |

tau = 3 U L / Re + 1/2. Below 0.501 the case needs allow below floor.

## Domain

| Setting | Meaning |
|---|---|
| nx, ny, nz | grid size; x is the flow direction, y up, z the span |
| x boundary | inflow (inlet and pressure outlet) or periodic |
| y boundary | walls, free_slip or periodic |
| floor, ceiling | static or moving (at U), with y walls |
| side walls | periodic or free_slip (z) |
| relax width x, z | absorbing layer thickness at the x ends and the z sides; 0 turns them off |
| relax sigma | layer strength |
| layer kind | fused (inside the regularized collision) or separate (needed for bgk and trt) |

## Models

| Setting | Meaning |
|---|---|
| sgs | none, wale (default) or smagorinsky |
| wale constant, smagorinsky constant | model constants; under WALE the Smagorinsky value is a floor |
| wall model | log-law wall model, staircase walls only |

## Timing

Run length is counted in flow-throughs, nx / U steps.

| Setting | Meaning |
|---|---|
| ramp, warmup, total flow throughs | start-up ramp, discarded warm-up, whole run |
| steps / warmup / ramp override | the same in steps, when given |
| sample every | steps between force samples |

## Geometry

| Kind | Settings |
|---|---|
| stl | path, cells per unit, offset |
| naca | four-digit section, chord, angle degrees, leading edge (x, y of the lowest point) |
| sphere | center, radius |
| cylinder | center (x, y), radius, spin ratio (ωR/U, positive anticlockwise about z) |
| ahmed | x start, body height, slant angle, nose (round or square); staircase walls only |

Every part has a name, a reference area for its coefficients, and wall: bouzidi (interpolated, default) or
staircase. An Ahmed body must fit the domain: its length is 3.6, its width 1.35 and its top 1.18 body heights
(plus one cell); the summary says which size does not fit.

## Resolution and placement

A wall is placed on the grid to a fraction of a cell, so forces should not change when a part moves by less
than a cell. They do when a curved edge is too tight for the grid: with a leading-edge radius of 1.3 cells
(NACA 4412 at chord 80) the lift changes by up to about 7% over a one-cell move, and by about 5% at 1.9 cells.
The part line warns when a part's smallest radius is under 5 cells. For a NACA section this is its leading
edge; for an STL, the tightest smoothly curved region (sharp edges are not counted).

- Where the coefficients matter, use a grid that puts at least 5 cells across the smallest radius.
- In a sweep over position, use steps of half a cell, or state the spread above with the results.
- Edges thinner than a cell, such as trailing edges, are handled by the walls and need no warning.

## Summary lines

- **resolved Re:** the Reynolds number the case runs at; capped where tau reaches its floor, unless allowed
  below it.
- **memory:** GPU memory of the solver's fields; red above 90% of the card.
- **run time:** cells x steps / throughput. The throughput is a typical value until a finished run on this
  machine measures it.

## Run folder

| File | Contents |
|---|---|
| case.json | the case exactly as run |
| solver.log | everything the solver printed |
| progress.json | live status and step |
| coefficients_live.csv | force coefficient samples per part |
| force_coefficients.npy | the same, as an array |
| result.json | status, per-part mean, standard error and drift of each coefficient, timings |
| record.csv | the run record |

Status is finished, stopped (Stop pressed) or blow_up (the solution diverged).

## Validated cases

Each template reproduces a reference run of the solver's validation record.

| Template | Result | Reference |
|---|---|---|
| wing ground | C_y -0.3987, C_x 0.1070 | release gate, every version (since 0.4.0; -0.3368 / 0.1144 before the wall-link fixes) |
| ahmed | Cd 0.6996 | Ahmed body, 25 degree slant, H = 32 |
| sphere | Cd 1.623 | Re 50; Schiller-Naumann 1.54, the gap is the domain confinement |
| spinning cylinder | Cd 1.163, Cl -2.656 | Re 100, spin ratio 1 (Magnus) |

## Known limits

- Relative comparisons and trends, not certification-grade absolute coefficients.
- Curved edges under about 5 cells in radius make forces depend on sub-cell placement (Resolution and
  placement); local grid refinement, planned, removes the need for a fine grid everywhere.
- High Reynolds numbers are capped by the grid; the case says when.
- The Ahmed body has staircase walls only.
- The rotating wheel problem type has no wheel body yet; the spinning cylinder template is the validated
  rotating case.
- Full list: DESIGN.md, known limits.
