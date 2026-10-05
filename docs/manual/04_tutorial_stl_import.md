# Tutorial: importing an STL

Any closed triangle mesh in STL format (binary or ASCII) can be a part.

## 1. Start a study

On the start page:

1. **Problem:** the card that matches the setup, for example Free stream.
2. **Study:** Averaged forces.
3. **Flow:** the real speed and body length; Resolution sets the cells across the body length.
4. **Geometry:** Import STL. The STL is scaled so its length in x is the body length in cells.

Press Create.

## 2. Check the import

The part's summary line shows its size in cells and in the STL's own units, with the scale between them:

```
body: 64.0 x 18.2 x 24.1 cells (4.2 x 1.2 x 1.6 units at 15.2 cells/unit)
```

- **Units:** a body far too small or too large in cells usually means millimeters read as meters or the
  reverse. Fix it with cells per unit on the part.
- **Under 10 cells across:** the part is too coarse to resolve.
- **Not watertight:** the mesh has holes, flipped triangles or edges shared by more than two triangles. The
  solver decides inside from outside by ray crossings, so an open mesh can fill or empty the wrong cells.
  Repair the mesh before relying on the result.
- **Extends outside the domain:** move it with the offset, or enlarge the domain.

An STL can also be added to an open case with **Home > Import STL**; it is then scaled to fit the domain.

## 3. Place it

The offset moves the part in cells after scaling. For a body on the ground, lower the y offset until the
lowest point sits the wanted distance above the floor (the floor's wall is at y = 0.5).

Check the reference area: it starts as the bounding-box frontal area, which overestimates most bodies. The
force coefficients divide by it.

## 4. Preview, save, run

Press **View > Voxels** and check the Voxels section for red text, then Save as a project. The STL is copied
into the project's geometry folder. Check the GPU memory line in the summary, then Run.
