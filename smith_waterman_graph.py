"""
Smith-Waterman Algorithm Implementation With Graphs
Robin Anwyl
December 2024
Description: Graph-based implementation of the Smith-Waterman algorithm for
local sequence alignment.
"""

import numpy as np
import pandas as pd


class Vertex:
    """
    A vertex in a graph, representing an entry in a scoring matrix.
    Attributes:
        _id: Tuple (row, col) of entry location in scoring matrix
        _score: Score of entry in scoring matrix
    """
    def __init__(self, vertex_id):
        self._id = vertex_id # Tuple (row, col)
        self._score = None # Initialize empty scoring matrix

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, new_id):
        self._id = new_id

    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, new_score):
        self._score = new_score

    def __str__(self):
        return f"{self._id}"

    def __repr__(self):
        return self.__str__()


class Graph:
    """
    A graph that represents the scoring matrix of the Smith-Waterman algorithm.
    Attributes:
        seq1 (str): First sequence to align
        seq2 (str): Second sequence to align
        nrows (int): Number of rows in scoring matrix = len(seq2)+1
        ncols (int): Number of columns in scoring matrix = len(seq1)+1
        vertices (list of lists): Array that holds the Vertex objects and
            represents the scoring matrix
        edges (dict): Adjacency list representation of graph that records
            edges between each source vertex ID and its destination vertex IDs.
        indegrees (dict): Number of incoming edges (indegree) of each vertex
        scores (NumPy array): NumPy array of scores in scoring matrix
        hi_score (float or int): Highest score in scoring matrix
        hi_v (list): List of Vertex objects with the highest score
        matrix (NumPy array): NumPy array representing scoring matrix with
            directional arrows
    Methods:
        add_edge: Adds an edge to the graph
        scoring_linear: Calculates scoring matrix with linear gap penalty
        display: Displays the scoring matrix
    """
    def __init__(self, seq1_, seq2_):
        """
        Initializes graph representing the scoring matrix for the alignment of
        seq1 and seq2. Initializes data structures for tracking vertices,
        edges, and scores. Sets row 0 and column 0 scores to 0.
        """
        # Initialize attributes
        self.seq1, self.seq2, = seq1_, seq2_
        self.nrows, self.ncols = len(seq1_) + 1, len(seq2_) + 1
        self.vertices = [[] for _ in range(self.nrows)]
        self.edges = dict()
        self.indegrees = dict()
        self.scores = np.full((self.nrows, self.ncols), np.nan)
        self.hi_score = float("-inf")
        self.hi_v = list()
        matrix_shape = 2*self.nrows+1, 2*self.ncols+1
        self.matrix = np.full(matrix_shape, " ", dtype="<U4")
        # Add vertices to graph
        for i in range(self.nrows):
            curr_row = self.vertices[i]
            for j in range(self.ncols):
                v = Vertex((i, j)) # Create vertex for (row, col)
                curr_row.append(v) # Add vertex to self.vertices array
        # Populate row 0
        for j in range(self.ncols):
            self.vertices[0][j].score = 0 # Set scores to 0
            self.edges[(0, j)] = list() # Add to self.edges
        self.scores[0, :] = 0
        # Populate column 0
        for i in range(1, self.nrows):
            self.vertices[i][0].score = 0
        self.scores[:, 0] = 0
        # Add row 0 and column 0 to self.edges
        for i in range(self.nrows):
            self.edges[(i, 0)] = list()
        for j in range(1, self.ncols):
            self.edges[(0, j)] = list()

    def add_edge(self, src_v, dst_v):
        """
        Adds a directed edge from the source vertex to the destination vertex.
        Updates self.edges and self.indegrees.
        """
        # Update self.edges adjacency list
        if src_v.id not in self.edges:
            # Add new source vertex
            self.edges[src_v.id] = [dst_v.id]
        else:
            # Update list of destination vertices for source vertex
            self.edges[src_v.id].append(dst_v.id)
        # Update self.indegrees
        if dst_v.id not in self.indegrees:
            # Add entry for destination vertex
            self.indegrees[dst_v.id] = 1
        else:
            # Increment indegree entry for destination vertex
            self.indegrees[dst_v.id] += 1

    def scoring_linear(self, match_score, mismatch_score, gap_penalty):
        """
        Generates sequence alignment scoring matrix using the Smith-Waterman
        algorithm with a linear gap penalty.
        """
        curr_hi_score = float("-inf")
        curr_hi_v = list()
        # Calculate all scores
        for i in range(1, self.nrows):
            char1 = self.seq1[i-1]
            for j in range(1, self.ncols):
                char2 = self.seq2[j-1]
                # Calculate similarity score between characters
                if char1 == char2:
                    sim_score = match_score
                else:
                    sim_score = mismatch_score
                # Calculate potential score of aligning char1 and char2
                score1 = self.scores[i-1, j-1] + sim_score
                # Calculate potential score if seq1 has a gap
                score2 = self.scores[i-1, j] - gap_penalty
                # Calculate potential score if seq2 has a gap
                score3 = self.scores[i, j-1] - gap_penalty
                # Calculate final score
                final_score = max(score1, score2, score3, 0)
                # Set score
                curr_v = self.vertices[i][j]
                curr_v.score = final_score
                self.scores[i, j] = final_score
                # Check for high score
                if final_score > curr_hi_score:
                    # Update new high score and high score vertex list
                    curr_hi_score = final_score
                    curr_hi_v = [curr_v]
                elif final_score == curr_hi_score:
                    # Update high score vertex list
                    curr_hi_v.append(curr_v)
                # If score is nonzero, add edge to graph
                if final_score == 0:
                    continue
                elif final_score == score1:
                    self.add_edge(self.vertices[i][j], self.vertices[i-1][j-1])
                elif final_score == score2:
                    self.add_edge(self.vertices[i][j], self.vertices[i-1][j])
                else:
                    self.add_edge(self.vertices[i][j], self.vertices[i][j-1])
        # Update final high score and high score vertices
        self.hi_score = int(curr_hi_score)
        self.hi_v = curr_hi_v
        # Cast self.scores to integer type
        self.scores = self.scores.astype(int)

    def display(self):
        """
        Displays the scoring matrix, with directional arrows included.
        """
        # Add rows and column numbers as headings
        row1 = [" "] + [_ for i in range(self.ncols) for _ in (" ", i)]
        col1 = [" "] + [_ for i in range(self.nrows) for _ in (" ", i)]
        self.matrix[0, :], self.matrix[:, 0] = row1, col1
        # Add sequences as row/column headings
        len1, len2 = len(self.seq1), len(self.seq2)
        sp3 = [" ", " ", " "]
        row2 = sp3 + [_ for i in range(len2) for _ in (" ", self.seq2[i])]
        col2 = sp3 + [_ for i in range(len1) for _ in (" ", self.seq1[i])]
        self.matrix[1, :], self.matrix[:, 1] = row2, col2
        # Add row/column 0 scores and arrows
        row3 = [char for _ in range(len2) for char in (0, "←")] + [0]
        col3 = [char for _ in range(len1) for char in (0, "↑")] + [0]
        self.matrix[2, 2:], self.matrix[2:, 2] = row3, col3
        # Add rest of scoring matrix and arrows
        self.matrix[4::2, 4::2] = self.scores[1:, 1:].astype(int)
        for v1 in self.edges:
            (i, j) = v1
            for v2 in self.edges[v1]:
                (x, y) = v2
                if x == i-1:
                    if y == j:
                        self.matrix[2*i+1][2*j+2] = "↑"
                    elif y == j-1:
                        self.matrix[2*i+1][2*j+1] = "↖"
                elif x == i and y == j-1:
                    self.matrix[2*i+2][2*j+1] = "←"

    def __str__(self):
        self.display()
        df = pd.DataFrame(self.matrix)
        return f"{df.to_string(index=False, header=False)}"

    def traceback(self, i, j, align1, align2, all_align):
        """
        Recursive traceback function.
        """
        # Base case: score = 0 (start of local alignment)
        if self.scores[i, j] == 0:
            align1, align2 = reversed(align1), reversed(align2)
            all_align.append(("".join(align1), "".join(align2)))
        pass


    def find_all_alignments(self):
        """
        Finds all possible tracebacks through scoring matrix from all vertices
        with the highest score.
        """
        pass

# Testing code
g = Graph("GGATCGA", "GAATTCAGTTA")
g.scoring_linear(5, -3, 4)
print(g)
print(g.edges)
