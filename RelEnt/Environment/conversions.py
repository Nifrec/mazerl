
def int_to_coord(s, grid_size):
    """
    Gets coordinate on the grid that corresponds to the given state
    :param s: int - state
    :param grid_size: int - side len of square grid
    :return: tuple(float, float) of coordinates
    """
    return s // grid_size, s % grid_size

def coord_to_int(c, grid_size):
    """
    Gets the state that corresponds to the given coordinates on the grid
    :param c: tuple(float, float) of coordinates
    :param grid_size: int - side len of square grid
    :return: int - state
    """
    return c[1] + c[0] * grid_size