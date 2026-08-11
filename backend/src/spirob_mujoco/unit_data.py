"""Unit geometry shared with the TypeScript simulator.

Ported verbatim from ``src/physics.ts`` (``UNIT_DATA``). Each row is the STL
connected-component analysis of the 3-cable SpiRob model:

    [x_center_mm, length_mm, height_mm, width_mm]

``x_center`` decreases toward the tip; the robot extends along +X in the
MuJoCo model with the base unit fixed to the world.

Axis convention: MuJoCo is Z-up while the Three.js frontend is Y-up. The
mapping is (x, y, z)_three = (x, z, -y)_mujoco and is applied only in the
transport layer, never inside the physics.
"""

MM = 0.001

# fmt: off
UNIT_DATA = [
    [-165.86769, 13.86136, 27.77303, 31.41718],
    [-183.25348, 19.05084, 26.06857, 29.34485],
    [-202.49631, 17.70167, 24.23258, 27.27514],
    [-220.37899, 16.44785, 22.52637, 25.3518],
    [-236.99764, 15.28265, 20.94077, 23.56439],
    [-252.44162, 14.19982, 19.46723, 21.90326],
    [-266.79398, 13.19355, 18.09785, 20.35956],
    [-280.13184, 12.25836, 16.82526, 18.92499],
    [-292.52693, 11.38931, 15.64262, 17.5918],
    [-304.04588, 10.58169, 14.54358, 16.35272],
    [-314.75065, 9.83115, 13.52222, 15.20136],
    [-324.69876, 9.13366, 12.57306, 14.13122],
    [-333.94368, 8.48544, 11.69098, 13.13678],
    [-342.53516, 7.88312, 10.87126, 12.21265],
    [-350.51935, 7.3233, 10.10946, 11.35376],
    [-357.9392, 6.80307, 9.40152, 10.55554],
    [-364.83454, 6.31961, 8.74361, 9.81372],
    [-371.24255, 5.87036, 8.13221, 9.12433],
    [-377.1976, 5.45282, 7.56401, 8.4837],
    [-382.73171, 5.06479, 7.03599, 7.88824],
]
# fmt: on

UNIT_COUNT = len(UNIT_DATA)


def along_m(i: int) -> float:
    """Distance of unit i's center from the base unit center, in meters."""
    return (UNIT_DATA[0][0] - UNIT_DATA[i][0]) * MM


def cable_radius_m(i: int) -> float:
    """Cable attachment radius on unit i's cross-section, in meters.

    Same convention as the rod solver: mean of height/width half-extents
    scaled by 0.76 to sit inside the printed wall.
    """
    return (UNIT_DATA[i][2] + UNIT_DATA[i][3]) * 0.25 * 0.76 * MM


def unit_volumes() -> list[float]:
    return [length * h * w for _, length, h, w in UNIT_DATA]
