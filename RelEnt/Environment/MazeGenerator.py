import random
import numpy as np

from collections import deque


class MazeGenerator:
    """
    Generates a grid maze, where starting point in col 0, end point - in last col
    """

    def __init__(self, grid_size):
        """
        :param grid_size: side len of the maze
        """
        self.grid_size = grid_size

        # Maze cells - one random connected path through cells from
        # self.maze_start_point to self.maze_end_point
        self.maze_cells = set()
        # Distraction cells - random non-connected paths through cells
        self.distraction_cells = set()

        # nr of random cells that are later connected with maze cells
        self.nr_maze_cells = self.grid_size // 2
        # nr of pairs of (start, end) cells of distraction paths
        self.nr_distraction_pairs = self.grid_size // 4

        self.maze_start_point = (0, 0)
        self.maze_end_point = (grid_size - 1, grid_size - 1)

    def find_neighbours(self, coord):
        """
        Finds the neighour of a cell
        :param coord: tuple coordinate
        :return: list of tuples of neighbours in grid
        """
        x, y = coord

        right = (x + 1, y)
        down = (x, y + 1)
        left = (x - 1, y)
        up = (x, y - 1)
        neighbouring_points = [right, down, left, up]

        if x == self.grid_size - 1:
            # If we are the right border
            neighbouring_points.remove(right)
        if x == 0:
            # If we are the left border
            neighbouring_points.remove(left)
        if y == self.grid_size - 1:
            # If we are at the bottom
            neighbouring_points.remove(down)
        if y == 0:
            # If we are at the top
            neighbouring_points.remove(up)

        return neighbouring_points

    def construct_graph(self):
        """
        Transforming grid into a graph with adjacency dict (for dijkstra)
        :return: adjacency_dict - dict(coord: neighbours)
                 vertices - list of all coords
                 edges - list of all edges (point, neighbour) with cost 1
                 dijkstra_dict - dict(coord: list(neighbour, cost))
        """
        adjacency_dict = {}
        vertices = []
        edges = []
        dijkstra_dict = {}

        for x in range(self.grid_size):
            for y in range(self.grid_size):

                point = (x, y)
                vertices.append(point)
                adjacency_dict[point] = self.find_neighbours(point)

                dijkstra_dict[point] = []
                for neighbour in adjacency_dict[point]:
                    n_x, n_y = neighbour
                    edges.append((point, neighbour, 1))

                    dijkstra_dict[point].append((n_x, n_y, 1))

        return adjacency_dict, vertices, edges, dijkstra_dict

    def random_cell(self):
        """
        :return: random point on the grid
        """
        return random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)

    def generate_random_cells(self):
        """
        Generating random cells for maze composition
        :return: maze_points - points to connect to get one path
            from self.maze_start_point to self.maze_end_point
                 distraction_pairs - list of pairs of endpoints to get paths
        """
        maze_points = []

        # Generating maze start coord (left wall)
        self.maze_start_point = (random.randint(0, self.grid_size - 1), 0)
        maze_points.append(self.maze_start_point)

        # Generating intermediate maze points
        while len(maze_points) < self.nr_maze_cells - 1:
            point = self.random_cell()
            if point not in maze_points:
                maze_points.append(point)

        # Generating maze end coord (right wall)
        self.maze_end_point = (random.randint(0, self.grid_size - 1), self.grid_size - 1)
        maze_points.append(self.maze_end_point)

        distraction_pairs = []

        # Generating distraction pairs
        while len(distraction_pairs) < self.nr_distraction_pairs:
            pair = (self.random_cell(), self.random_cell())
            distraction_pairs.append(pair)

        return maze_points, distraction_pairs

    def generate_paths(self):
        """
        Connecting all the points to get paths to fill the maze with
        """

        adjacency_dict, vertices, edges, dijkstra_dict = self.construct_graph()
        maze_points, distraction_pairs = self.generate_random_cells()

        #print("MAZE POINTS = ", maze_points)

        # Connecting maze cells with dijkstra
        for i in range(len(maze_points) - 1):
            start = maze_points[i]
            end = maze_points[i + 1]
            shortest_path_maze = self.dijkstra(vertices, edges, dijkstra_dict,
                                               start, end)

            for state in shortest_path_maze:
                self.maze_cells.add(state)

        #print("MAZE CELLS = ", self.maze_cells)
        #print("DISTRACTION PAIRS = ", distraction_pairs)

        # Connecting each pair of distraction cells with dijkstra
        for pair in distraction_pairs:
            start = pair[0]
            end = pair[1]
            shortest_path_pair = self.dijkstra(vertices, edges, dijkstra_dict,
                                               start, end)

            for state in shortest_path_pair:
                self.distraction_cells.add(state)

        #print("DISTRACTION CELLS = ", self.distraction_cells)

    def generate_maze(self):
        """
        Generates the maze and reward matrix
        :return: matrix of rewards
        """
        self.generate_paths()

        # Start with all rewards = -1
        matrix = np.full((self.grid_size, self.grid_size), -1, dtype=int)

        # If a cell is part of either maze of distraction paths, reward = 0
        for x, y in self.maze_cells:
            matrix[x][y] = 0
        for x, y in self.distraction_cells:
            matrix[x][y] = 0

        # If cell = self.maze_end_point, reward = 1
        matrix[self.maze_end_point[0]][self.maze_end_point[1]] = 1

        return matrix

    def dijkstra(self, vertices, edges, dijkstra_dict, start, end):
        """
        Performs dijkstra algorithm
        :param vertices: list
        :param edges: list
        :param dijkstra_dict: dict(coord: list(neighbour, cost))
        :param start: star node of path
        :param end: end node of path
        :return: shortest path between star and end
        """
        assert start in vertices, 'Such start node doesn\'t exist'

        # Mark all nodes unvisited
        # Set the all distances to inf
        distances = {vertex: float('inf') for vertex in vertices}

        # Mark all nodes unvisited
        previous_vertices = {
            vertex: None for vertex in vertices
        }

        # Set the distance to zero for our initial node
        distances[start] = 0
        unvisited_vertices = vertices.copy()

        while unvisited_vertices:

            # Select the unvisited node with the smallest distance,
            current_vertex = min(unvisited_vertices,
                                 key=lambda vertex: distances[vertex])

            # Stop if the smallest distance among the unvisited nodes is inf
            if distances[current_vertex] == float('inf'):
                break

            # Find unvisited neighbors for the current node
            # and calculate their distances through the current node
            for neighbour_x, neighbour_y, cost in dijkstra_dict[current_vertex]:
                neighbour = (neighbour_x, neighbour_y)
                alternative_route = distances[current_vertex] + cost

                # Compare the newly calculated distance to the assigned
                # and save the smaller one
                if alternative_route < distances[neighbour]:
                    distances[neighbour] = alternative_route
                    previous_vertices[neighbour] = current_vertex

            # Mark the current node as visited by removing from unvisited
            unvisited_vertices.remove(current_vertex)

        path, current_vertex = deque(), end
        while previous_vertices[current_vertex] is not None:
            path.appendleft(current_vertex)
            current_vertex = previous_vertices[current_vertex]
        if path:
            path.appendleft(current_vertex)

        return path