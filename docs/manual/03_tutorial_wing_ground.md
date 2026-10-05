# Tutorial: wing in ground effect

A NACA 4412 section at 4 degrees, close to a moving ground, in a thin span (quasi-2D). This is the case every
release is checked against.

## 1. Open the template

On the start page, double-click **wing ground** under Templates. The case opens untitled.

The summary should read Ready to run, with tau 0.50240, Mach 0.087 and 1,280,000 cells. The domain is
800 x 400 x 4 cells; the wing chord is 80 cells and its lowest point sits 24.5 cells above the floor.

## 2. Check the geometry

Press **View > Voxels**. The geometry is built exactly as the solver will build it. The window is locked
until it is ready.

- The cubes are the solid cells.
- The dots are the Bouzidi wall points, where each link meets the true surface, colored by the wall fraction
  q from 0 to 1.
- The Voxels section of the summary lists the solid cells and boundary links. Red text there means a wall
  would not work as intended.

Any edit clears the preview.

## 3. Save and run

Save as a project, for example `wing ground 1`, then press **Run** (F5). The Results tab opens:

- The shaded region is the warm-up, discarded from the averages.
- The thin lines are the force coefficients; the thick lines are their running means.

The run takes 10 flow-throughs (160,000 steps), about ten minutes on a recent GPU; the summary's run-time
line gives the estimate for this machine.

## 4. Read the result

The Last run summary should show:

| | Expected |
|---|---|
| C_y (downforce, negative) | -0.3368, standard error 0.0004 |
| C_x (drag) | 0.1144 |

The run folder holds the full record (Reference, run folder).

## 5. Next steps

Change the ride height (Geometry > wing > leading edge, second value) or the angle (angle degrees), save,
and run again. Comparing those runs is what the solver is built for.
