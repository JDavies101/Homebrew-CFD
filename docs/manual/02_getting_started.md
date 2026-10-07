# Getting started

## Start page

The app opens on the start page, unless it was started with a case file.

- **Steps on the left:** Problem, Study, Flow, Geometry. Visited steps can be clicked again; a check mark
  shows while the answers give a valid case.
- **Recent cases and Templates** under the steps: double-click to open. A template opens untitled, so saving
  never overwrites it.
- **Preview and verdict on the right:** the domain and body as the answers describe them, with the physical
  and resolved Reynolds number, tau, Mach, grid size, GPU memory and an estimated run time.
- **Bottom bar:** Open case, Blank case, Back, Next. On the last step Next becomes Create.

Only Problem, Study and Flow need answers; every field has a default. More options opens the extra settings
on a step. Create opens the case for review; it never starts a run.

## Projects

Save as asks for a project name and creates one folder for everything that belongs to the case:

```
Documents\Homebrew CFD Projects\
  <project>\
    <project>.json    the case
    geometry\         imported STLs, copied in
    runs\             one folder per run
    exports\
```

The case stores its STLs relative to itself, so a project folder can be moved, zipped or copied to another
machine whole. Cases saved before projects existed still open and run; their runs go to the shared Runs
folder.

## Main window

- **Home:** New from a template, Open, Save, Save as; Import STL, Remove part. Ctrl+N opens the start page.
- **View:** Fit, Orthographic, the standard views; Voxels (the geometry preview); Reset layout.
- **Run:** Run (F5) runs the selected study; Stop (Shift+F5); New study.

The left half holds the tree and the settings of the selected node:

- **Case:** Flow, Domain, Models, Geometry (one node per part). A part shows only the settings its kind uses.
- **Study:** one node per study. A study is one way of solving the case: its type, its solver and timing
  settings, and an optional parametric sweep below it.
- **Results:** one node per study that has run: its force coefficients, or the sweep plot of a parametric
  study.

The right half shows the selected item (the case in cells, a run's coefficients, or a sweep plot) above three
tabs: Summary (the verdict, derived numbers, the estimate, every part's size and warnings, the last run),
Console (what the solver prints) and Queue (once a parametric sweep has run).

Inputs are locked while a run or a geometry preview is in progress; the tree stays navigable. Save on a case
that has never been saved asks for a name, like Save as.

## Studies

Right-click **Study > New study** (or Run > New study). A new study copies the case's solver and timing; edit
them on the study's node. Run (F5) or **Run study** writes the study to the project's `studies\` folder and runs
it. A study that has run shows **Run again**: it runs the study with its current settings and replaces its
earlier results, after a confirmation.

A parametric sweep runs the same study once per value. Right-click a study that has not run yet and choose
**Add parametric sweep**, then on the sweep's node pick the part, the parameter (position y in cells, or
angle in degrees) and the values, as a list or as start, end and count. The node shows the number of runs and
the total time. The runs queue one after another; the Results node of the study plots C_x and C_y against
the swept value.

## Queue

The Queue tab lists the runs waiting and the one running. Waiting runs can be removed or moved; Clear queue
drops them all. Pause lets the current run finish and then holds the queue; Start/Resume continues it. Stop
ends the run in progress and the rest of its study.

Stop needs two clicks within three seconds (the button turns to Stop?), on the ribbon or on the square next
to the progress bar.
