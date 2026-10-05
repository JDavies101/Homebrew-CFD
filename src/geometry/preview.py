# geometry preview: solid mask and boundary links of built parts, saved for the app to draw without Taichi
from pathlib import Path
import numpy as np
from src.engine import lattice_d3q19 as d3q19

def boundary_links(wall_fractions):
    """
    Every Bouzidi boundary link of a part: fluid node, direction and wall fraction (q > 0 marks a link).

    Returns nodes (L, 3) int32, directions (L,) int8, fractions (L,) float32.
    """

    directions, i, j, k = np.nonzero(wall_fractions > 0.0)
    nodes = np.stack([i, j, k], axis=1).astype(np.int32)

    return nodes, directions.astype(np.int8), wall_fractions[directions, i, j, k].astype(np.float32)

def wall_points(nodes, directions, fractions):
    """
    Where each link meets the wall: x + q c_q.

    Returns points (L, 3) float32.
    """

    return (nodes + fractions[:, None] * d3q19.lattice_velocities[directions]).astype(np.float32)

def save_preview(path, parts, shape):
    """
    Write one .npz: part id per cell (0 = fluid) and every Bouzidi part's links with their part id.
    """

    part_id = np.zeros(shape, np.uint8)
    nodes = [np.zeros((0, 3), np.int32)]
    directions = [np.zeros(0, np.int8)]
    fractions = [np.zeros(0, np.float32)]
    link_part = [np.zeros(0, np.uint8)]

    for index, part in enumerate(parts):
        part_id[part.solid > 0] = index + 1

        if part.wall_fractions is not None:
            part_nodes, part_directions, part_fractions = boundary_links(part.wall_fractions)
            nodes.append(part_nodes)
            directions.append(part_directions)
            fractions.append(part_fractions)
            link_part.append(np.full(len(part_fractions), index + 1, np.uint8))

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, part_id=part_id, nodes=np.concatenate(nodes),
                        directions=np.concatenate(directions),
                        fractions=np.concatenate(fractions), link_part=np.concatenate(link_part),
                        names=np.array([part.name for part in parts], dtype=str),
                        bouzidi=np.array([part.wall_fractions is not None for part in parts],
                                         dtype=bool))
    
def load_preview(path):
    """
    Read a preview file into plain arrays (no pickled objects).

    Returns a dict of arrays.
    """

    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}