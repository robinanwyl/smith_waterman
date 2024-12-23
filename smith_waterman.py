"""
Smith-Waterman Algorithm Implementation
Robin Anwyl
December 2024
Description: Implementation of the Smith-Waterman algorithm for local sequence
alignment.
"""

class Vertex:
    """
    A vertex in a graph, representing an entry in a scoring matrix.
    Attributes:
        id: Tuple (row, col) of entry location in scoring matrix
        score: Score of entry in scoring matrix
    """
    def __init__(self, id_):
        self.id = id_ # Tuple (row, col)
        self.score = "empty" # Initialize empty scoring matrix

    @property
    def score(self):
        return self.score

    @score.setter
    def score(self, new_score):
        self.score = new_score

    def __str__(self):
        return f"{self.id}"

    def __repr__(self):
        return self.__str__()


class Graph:
    """
    A graph that represents the scoring matrix of the Smith-Waterman algorithm.
    Attributes:
        nrows (int): number of rows in scoring matrix = len(seq2)+1
        ncols (int): number of columns in scoring matrix = len(seq1)+1
        vertices (list): List of lists (array) that holds the Vertex objects
            and represents the scoring matrix
        edges (dict): Adjacency list of graph representing all edges
        num_vertices (int): Number of vertices (recorded for testing)
        num_edges (int): Number of edges (recorded for testing)
    Methods:
        add_edge: Adds an edge to the graph
        static method iter_: Iterates through a dict where each value is a list
    """
    def __init__(self, seq1, seq2):
        """
        Initializes graph representing the scoring matrix for the alignment of
        two given sequences (seq1 and seq2). Stores the graph Vertex objects in
        the array self.vertices. Sets scores of entries in row 0 and column 0
        to 0.
        """
        self.nrows = len(seq2) + 1
        self.ncols = len(seq1) + 1
        self.vertices = list()
        self.edges = dict()
        self.num_vertices = 0
        self.num_edges = 0
        # Add vertices to graph
        for i in range(self.nrows):
            self.vertices.append([]) # Add row to self.vertices
            curr_row = self.vertices[i]
            for j in range(self.ncols):
                v = Vertex((i, j)) # Create vertex for (row, col)
                curr_row.append(v) # Add vertex to self.vertices array
                self.num_vertices += 1 # Increment vertex count
        # Set row 0 scores to 0
        for j in self.vertices[0]:
            self.vertices[0][j].score = 0
        # Set column 0 scores to 0
        for i in range(self.nrows):
            self.vertices[i][0].score = 0

    def add_edge(self, src_v, dst_v):
        """
        Adds an edge, that connects two vertices, to the graph - specifically
        by updating the self.edges adjacency list representation of the graph.
        """
        if src_v.id not in self.edges:
            self.edges[src_v.id] = [dst_v.id]
        else:
            self.edges[src_v.id].append(dst_v.id)

    def __str__(self):
        return "\n".join([f"{e.src.id} -> {e.dst.id}" for e in
                          Graph.iter_(self.edges)])

    @staticmethod
    def iter_(dict_of_lists):
        """
        Iterates through all items in a dict where each value is a list, such
        as self.edges.
        """
        for list_ in dict_of_lists:
            for item in dict_of_lists[list_]:
                yield item
