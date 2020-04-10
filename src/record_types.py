"""
File providing record types of elementary model-specific objects
used by Model and MazeLayout.

Author: Lulof Pirée
"""
import numpy as np
from numbers import Number

class Size():
    """
    Record type for width & height of rectangular coordinate planes.
    """

    def __init__(self, x: Number, y: Number):
        self.x = x
        self.y = y

    def copy(self):
        return Size(self.x, self.y)

class Line():
    """
    Record type for a (finite) straight line segment 
    in Cartesian coordinate space.
    """

    def __init__(self, x1: Number, y1: Number, x2: Number, y2: Number):
        self.p0 = np.array([x1, y1])
        self.p1 = np.array([x2, y2])
        #self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Coefficients of equation ax + by + c = 0 for this line.
        

    def __str__(self):
        return f"Line ({self.p0}, {self.p1})"

class Ball():
    """
    Record type for a moving ball, has a position, radius,
    velocity (in 2 dimensions) and acceleration (in 2 dimensions).
    """
    def __init__(self, x: Number, y: Number, rad: Number, x_vel: Number=0,
            y_vel: Number=0, x_acc: Number=0, y_acc: Number=0):
        self.pos = np.array([x, y])
        self.rad = rad
        self.vel = np.array([x_vel, y_vel])
        self.acc = np.array([x_acc, y_acc])