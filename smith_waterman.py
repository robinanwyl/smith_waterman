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
        seq1 (str): First sequence to align
        seq2 (str): Second sequence to align
        nrows (int): Number of rows in scoring matrix = len(seq2)+1
        ncols (int): Number of columns in scoring matrix = len(seq1)+1
        vertices (list): List of lists (array) that holds the Vertex objects
            and represents the scoring matrix
        edges (dict): Adjacency list representation of graph that records
            edges between each source vertex and its destination vertices.
        hi_score: Highest score in scoring matrix
        hi_v (list): List of ids of vertices with the highest scores
        num_vertices (int): Number of vertices (recorded for testing)
        num_edges (int): Number of edges (recorded for testing)
    Methods:
        add_edge: Adds an edge to the graph
        scoring_linear: Calculates scoring matrix with linear gap penalty
        static method iter_: Iterates through a dict where each value is a list
    """
    def __init__(self, seq1_, seq2_):
        """
        Initializes graph representing the scoring matrix for the alignment of
        two given sequences (seq1 and seq2). Stores the graph Vertex objects in
        the array self.vertices. Sets scores of entries in row 0 and column 0
        to 0.
        """
        self.seq1 = seq1_
        self.seq2 = seq2_
        self.nrows = len(seq2_) + 1
        self.ncols = len(seq1_) + 1
        self.vertices = list()
        self.edges = dict()
        self.hi_score = float("-inf")
        self.hi_v = list()
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
        Adds a directed edge from the source vertex to the destination vertex.
        Updates the self.edges adjacency list representation of the graph.
        """
        if src_v.id not in self.edges:
            # Add new source vertex
            self.edges[src_v.id] = [dst_v.id]
        else:
            # Update list of destination vertices for source vertex
            self.edges[src_v.id].append(dst_v.id)
        self.num_edges += 1 # Increment edge count

    def scoring_linear(self, match_score, mismatch_score, gap_penalty):
        """
        Generates sequence alignment scoring matrix using the Smith-Waterman
        algorithm with a linear gap penalty.
        """
        curr_hi_score = float("-inf")
        curr_hi_v = list()
        for i in range(1, self.nrows):
            for j in range(1, self.ncols):
                # Calculate similarity score between characters
                char1 = self.seq1[i-1]
                char2 = self.seq2[j-1]
                if char1 == char2:
                    sim_score = match_score
                else:
                    sim_score = mismatch_score
                # Calculate score of aligning char1 and char2
                score1 = self.vertices[i-1][j-1].score + sim_score
                # Calculate score if seq1 has a gap
                score2 = self.vertices[i-1][j].score - gap_penalty
                # Calculate score if seq2 has a gap
                score3 = self.vertices[i][j-1].score - gap_penalty
                final_score = max(score1, score2, score3, 0) # Calculate score
                curr_v = self.vertices[i][j] # Current vertex
                curr_v.score = final_score # Set score
                # Check for high score
                if final_score > curr_hi_score:
                    # Update new high score and high score vertex list
                    curr_hi_score = final_score
                    curr_hi_v = [curr_v.id]
                elif final_score == curr_hi_score:
                    # Update high score vertex list
                    curr_hi_v.append(curr_v.id)
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
        self.hi_score = curr_hi_score
        self.hi_v = curr_hi_v
        return None

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
