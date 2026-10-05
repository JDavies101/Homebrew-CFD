# Reference

Lengths are in cells, velocities and times in lattice units (one step, one cell), unless marked.

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
staircase.

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
| wing ground | C_y -0.3368, C_x 0.1144 | release gate, every version |
| ahmed | Cd 0.6996 | Ahmed body, 25 degree slant, H = 32 |
| sphere | Cd 1.623 | Re 50; Schiller-Naumann 1.54, the gap is the domain confinement |
| spinning cylinder | Cd 1.164, Cl -2.652 | Re 100, spin ratio 1 (Magnus) |

## Known limits

- Relative comparisons and trends, not certification-grade absolute coefficients.
- High Reynolds numbers are capped by the grid; the case says when.
- The Ahmed body has staircase walls only.
- The rotating wheel problem type has no wheel body yet; the spinning cylinder template is the validated
  rotating case.
- Full list: DESIGN.md, known limits.
