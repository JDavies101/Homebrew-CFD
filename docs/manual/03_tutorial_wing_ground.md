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

The wing's part line also warns that its smallest radius, the leading edge, is about 1.3 cells (Reference,
resolution and placement). That is expected for this template.

## 3. Save and run

Save as a project, for example `wing ground 1`, then press **Run** (F5). With no study yet, Study 1 is created
from the case's settings and runs. The tree selects Results > Study 1: force coefficients:

- The shaded region is the warm-up, discarded from the averages.
- The thin lines are the force coefficients; the thick lines are their running means.

The run takes 10 flow-throughs (160,000 steps), about five to ten minutes on a recent GPU; the summary's
run-time line gives the estimate for this machine.

## 4. Read the result

The Last run summary should show:

| | Expected |
|---|---|
| C_y (downforce, negative) | -0.3987, standard error 0.0004 |
| C_x (drag) | 0.1070 |

The run folder holds the full record (Reference, run folder).

## 5. Ride-height sweep

Right-click **Study > New study**, then right-click the new study and choose **Add parametric sweep**. On the
sweep's node:

- Part: wing. Parameter: Position y (cells).
- Values: `120.5, 48.5, 24.5, 12.5, 8.5` (h/c 1.5, 0.6, 0.3, 0.15, 0.1; y = 80 h/c + 0.5).

Run the study. The five runs queue one after another, and its Results node plots C_x and C_y against the
lowest-point height: the ground-effect trend of this wing. Moves in steps of half a cell keep sub-cell
placement from mixing into the trend (Reference, resolution and placement).
