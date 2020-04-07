"""
Model providing the maze-simulation of the environment.

Author: Lulof Pirée
"""
import math


class Size():
    """
    Record type for width & height of rectangular coordinate planes.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Point():
    """
    Record type for a point in a Cartesian plane.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Line():
    """
    Record type for a straight line in Cartesian coordinate space.
    """

    def __init__(self, x1, y1, x2, y2):
        self.p1 = Point(x1, y1)
        self.p2 = Point(x2, y2)
        #self.length = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        # Coefficients of equation ax + by + c = 0 for this line.
        # Proof:
        # [x2]   [x1-x2]          [x]
        # [y2] + [y1-y2]*lambda = [y]
        # 
        # --> lambda = (x - x2)/(x1-x2)
        # --> y = y2 + (y1-y2)*lambda = y2 + (y1-y2)(x - x2)/(x1-x2)
        # --> y(x1 - x2) + x(y2 - y1) + y2(x2 - x1) +x2(y1 - y2) = 0
        #self.a = y2 - y1
        #self.b = x1 - x2
        #self.c = y2*(x2 - x1) + x2*(y1 - y2)

    def __str__(self):
        return f"Line ({self.p1.x}, {self.p1.y}) -> ({self.p2.x}, {self.p2.y})"

class Ball():
    """
    Record type for a moving ball, has a position, radius,
    velocity (in 2 dimensions) and acceleration (in 2 dimensions).
    """
    def __init__(self, x, y, rad, x_vel=0, y_vel=0, x_acc=0, y_acc=0):
        self.x = x
        self.y = y
        self.rad = rad
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.x_acc = x_acc
        self.y_acc = y_acc

class MazeLayout():

    def __init__(self, lines=set()):
        assert(isinstance(lines, set))
        self.lines = lines

    def get_ball_hits_wall(self, ball):
        assert(isinstance(ball, Ball))

        for line in self.lines:
            if (self.__collides(ball, line)):
                return True
        return False

    def __collides(self, ball, line):
        """
        Checks if a given ball collides with a given line.
        Note that just hitting without overlapping counts as hitting.
        Also note that lines have a finite length here.
        """
        # First check if the distance of the center of the ball to the line
        # is smaller than the radius of the ball.
        # Proof: https://brilliant.org/wiki/dot-product-distance-between-point-and-a-line/
        distance = abs(line.a * ball.x + line.b * ball.y + line.c ) \
                / math.sqrt(line.a**2 + line.b**2)
        if (((line.x1 < ball.x) == (line.x2 < ball.x)) and
                ((line.y1 < ball.y) == (line.y2 < ball.y))):
                return False
        
        if (distance <= ball.rad):
            return True
        return False


        # 

