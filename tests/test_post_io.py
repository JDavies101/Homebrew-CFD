# post-processing I/O smoke tests: VTK file written, plots return figures, progress bar output
import matplotlib
matplotlib.use("Agg")
import numpy as np
from src.post.vtk import write_field
from src.post import plotting
from src.post.progress import Progress

rng = np.random.default_rng(0)
# test 1: write_field creates the .vti file (and its folder)
def test_write_field_creates_file(tmp_path):

    density = np.ones((4, 5, 6), np.float32)
    velocity = rng.uniform(-0.1, 0.1, (3, 4, 5, 6)).astype(np.float32)
    path = tmp_path / "fields" / "sample"
    write_field(str(path), density, velocity)
    written = tmp_path / "fields" / "sample.vti"

    assert written.exists() and written.stat().st_size > 0

# test 2: every plot function returns a figure with the expected axes
def test_plot_functions_return_figures():

    velocity_2d = rng.uniform(-0.1, 0.1, (2, 8, 6))
    velocity_3d = rng.uniform(-0.1, 0.1, (3, 8, 6, 4))
    solid = np.zeros((8, 6, 4), np.int32)
    y_plus = np.linspace(1.0, 100.0, 20)
    figures = [
        plotting.plot_velocity_magnitude(velocity_2d)[0],
        plotting.plot_streamlines(velocity_2d)[0],
        plotting.plot_law_of_wall(y_plus, np.log(y_plus) / 0.41 + 5.2, y_plus * 0 + 1, y_plus * 0 + 1, y_plus * 0 + 1)[0],
        plotting.plot_mask_slice(solid, axis=2, index=1)[0],
        plotting.plot_velocity_slice(velocity_3d, axis=2, index=1, component=0)[0],
    ]

    assert all(isinstance(figure, matplotlib.figure.Figure) for figure in figures)
    assert len(figures[2].axes) == 2  # law of the wall: mean and rms panels

# test 3: progress bar reports the step count and finishes at 100%
def test_progress_output(capsys):

    progress = Progress(10)
    progress.update(4, health=0.05)
    progress.done()
    output = capsys.readouterr().out

    assert "5/10" in output and "max |u|=0.05" in output
    assert "10/10 100.0%" in output
