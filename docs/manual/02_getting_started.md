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

- **Home:** New study (Ctrl+N), New from a template, Open, Save, Save as; Import STL, Remove part.
- **View:** Fit, Orthographic, the standard views; Voxels (the geometry preview); Console, Reset layout.
- **Run:** Run (F5), Stop (Shift+F5).
- **Results:** the coefficient monitors of the current or last run.

The left pane is the setup tree (Solver, Flow, Domain, Models, Timing, Geometry). The middle pane edits the
selected node. The right pane shows the case in cells and the summary: the verdict, derived numbers, the
estimate, every part's size and warnings, and the last run.

Inputs are locked while a run or a geometry preview is in progress; the tree stays navigable.
