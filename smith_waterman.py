"""
Smith-Waterman Algorithm Implementation
Robin Anwyl
December 2024
Description: NumPy array-based implementation of the Smith-Waterman algorithm
for local sequence alignment.
"""

import numpy as np
import pandas as pd


class ScoringMatrix:
    """
    A class that represents the scoring matrix of the Smith-Waterman algorithm.
    Attributes:
        seq1: First sequence to align
        seq2: Second sequence to align
        match_score: Match score (positive int)
        mismatch_score: Mismatch (negative int)
        gap_penalty: Gap penalty (positive int)
        nrows: Number of rows in scoring matrix = len(seq2)+1
        ncols: Number of columns in scoring matrix = len(seq1)+1
        scores: NumPy array of scoring matrix
        disp_matrix: NumPy array of scoring matrix with directional arrows and
            row/column headings, for printing
    Methods:
        score_linear(): Calculates scoring matrix with linear gap penalty
        traceback(): Backtraces through the scoring matrix to find local
            alignments from a given high score cell
        find_all_alignments(): Finds all optimal alignments of seq1 and seq2
        display(): Displays the scoring matrix with directional arrows
    """
    def __init__(self, seq1_, seq2_, match_, mismatch_, gap_penalty_):
        """
        Populates scoring matrix and display matrix.
        """
        self.seq1, self.seq2, = seq1_, seq2_
        self.match_score, self.mismatch_score, self.gap_penalty = \
            (match_, mismatch_, gap_penalty_)
        self.nrows, self.ncols = len(seq1_) + 1, len(seq2_) + 1
        self.scores = np.full((self.nrows, self.ncols), np.nan)
        self.disp_matrix = np.full((2*self.nrows+1, 2*self.ncols+1),
                                    " ", dtype="<U5")
        # Populate row 0 and column 0 scores
        self.scores[0, :] = 0
        self.scores[:, 0] = 0
        # Calculate scores
        self.score_linear()
        # Generate display matrix
        self.display()

    def score_linear(self):
        """
        Generates scoring matrix with linear gap penalty.
        """
        for i in range(1, self.nrows):
            char1 = self.seq1[i-1]
            for j in range(1, self.ncols):
                char2 = self.seq2[j-1]
                # Calculate similarity score between characters
                if char1 == char2:
                    sim_score = self.match_score
                else:
                    sim_score = self.mismatch_score
                # Calculate potential score of aligning char1 and char2
                score1 = self.scores[i-1, j-1] + sim_score
                # Calculate potential score if seq1 has a gap
                score2 = self.scores[i-1, j] - self.gap_penalty
                # Calculate potential score if seq2 has a gap
                score3 = self.scores[i, j-1] - self.gap_penalty
                # Calculate final score
                final_score = max(score1, score2, score3, 0)
                # Set score
                self.scores[i, j] = final_score
        # Cast self.scores to integer type
        self.scores = self.scores.astype(int)

    def traceback(self, i, j, align1, align2, aligns):
        """
        Traceback through the scoring matrix from a high score cell. Helper
        method for find_all_alignments().
        :param i: Row number of current cell
        :param j: Column number of current cell
        :param align1: Partial alignment of seq1
        :param align2: Partial alignment of seq2
        :param aligns: All alignments from starting cell
        """
        pass

    def find_all_alignments(self):
        """
        Finds all optimal local alignments from the scoring matrix. Backtraces
        through the scoring matrix from all high scoring cells.
        """
        pass

    def display(self):
        """
        Displays the scoring matrix with directional arrows and row/column
        headings.
        """
        # Add rows and column numbers as headings
        row1 = [" "] + [_ for i in range(self.ncols) for _ in (" ", i)]
        col1 = [" "] + [_ for i in range(self.nrows) for _ in (" ", i)]
        self.disp_matrix[0, :], self.disp_matrix[:, 0] = row1, col1
        # Add sequences as row/column headings
        len1, len2 = len(self.seq1), len(self.seq2)
        sp3 = [" ", " ", " "]
        row2 = sp3 + [_ for i in range(len2) for _ in (" ", self.seq2[i])]
        col2 = sp3 + [_ for i in range(len1) for _ in (" ", self.seq1[i])]
        self.disp_matrix[1, :], self.disp_matrix[:, 1] = row2, col2
        # Add row/column 0 scores
        row3 = [char for _ in range(len2) for char in (0, " ")] + [0]
        col3 = [char for _ in range(len1) for char in (0, " ")] + [0]
        self.disp_matrix[2, 2:], self.disp_matrix[2:, 2] = row3, col3
        # Add rest of scoring matrix and arrows
        self.disp_matrix[4::2, 4::2] = self.scores[1:, 1:]
        for i in range(1, self.nrows):
            for j in range(1, self.ncols):
                if self.scores[i,j] == self.scores[i-1,j-1] + self.match_score:
                    self.disp_matrix[2*i+1, 2*j+1] = "↖"
                if self.scores[i,j] == self.scores[i-1, j] - self.gap_penalty:
                    self.disp_matrix[2*i+1, 2*j+2] = "↑"
                if self.scores[i,j] == self.scores[i, j-1] - self.gap_penalty:
                    self.disp_matrix[2*i+2, 2*j+1] = "←"

    def __str__(self):
        df = pd.DataFrame(self.disp_matrix)
        return f"{df.to_string(index=False, header=False)}"


# Testing code
m = ScoringMatrix("GGATCGA", "GAATTCAGTTA", 5, -3, 4)
print(m)