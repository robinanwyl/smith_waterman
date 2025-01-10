"""
Smith-Waterman Algorithm Implementation
Robin Anwyl
December 2024
Description: NumPy array-based implementation of the Smith-Waterman algorithm
for local sequence alignment, using a linear gap penalty.
"""

import numpy as np
import pandas as pd

class SmithWatermanLinear:
    """
    Implement the Smith-Waterman algorithm to perform local sequence alignment,
    using a linear gap penalty. (For best results, the given sequences should
    not contain gaps.)
    Attributes:
        seq1: First sequence to align
        seq2: Second sequence to align
        match_score: Match score (positive int)
        mismatch_score: Mismatch (negative int)
        gap_penalty: Gap penalty (positive int)
        nrows: Number of rows in scoring matrix = len(seq2)+1
        ncols: Number of columns in scoring matrix = len(seq1)+1
        scores: NumPy array of scoring matrix
        disp_scores: NumPy array of scoring matrix with directional arrows and
            row/column headings, formatted for display.
        high_score: Highest score in scoring matrix
        alignments: List of optimal local alignments
        disp_alignments: NumPy array of optimal local alignments, formatted
            for display.
    Methods:
        score_linear(): Perform scoring with linear gap penalty
        traceback(): Perform traceback through the scoring matrix, starting
            from a given high score cell
        find_all_alignments(): Find all optimal alignments of seq1 and seq2
        sequence_alignment(): Calculate scoring matrix, find all optimal
            alignments, and format scoring matrix and alignments for display
    """
    def __init__(self, seq1_: str, seq2_: str, match_: int, mismatch_: int,
                 gap_penalty_: int):
        """
        Initialize class attributes and perform initial setup of self.scores
        (scoring matrix) and self.disp_scores (scoring matrix formatted for
        display and used by score_linear() method).
        """
        self.seq1, self.seq2, = seq1_, seq2_
        self.match_score, self.mismatch_score, self.gap_penalty = \
            (match_, mismatch_, gap_penalty_)
        self.nrows, self.ncols = len(seq1_)+1, len(seq2_)+1
        self.scores = np.full((self.nrows, self.ncols), 0)
        self.disp_scores = np.full((2*self.nrows+1, 2*self.ncols+1)," ", dtype="<U5")
        self.high_score = 0
        self.alignments = list()
        self.disp_alignments = list()

        # Initial setup of self.disp_scores
        # Add rows and column numbers as headings
        row1 = [" "] + [_ for i in range(self.ncols) for _ in (" ", i)]
        col1 = [" "] + [_ for i in range(self.nrows) for _ in (" ", i)]
        self.disp_scores[0, :], self.disp_scores[:, 0] = row1, col1
        # Add sequences as row/column headings
        len1, len2 = len(self.seq1), len(self.seq2)
        sp3 = [" ", " ", " "]
        row2 = sp3 + [_ for i in range(len2) for _ in (" ", self.seq2[i])]
        col2 = sp3 + [_ for i in range(len1) for _ in (" ", self.seq1[i])]
        self.disp_scores[1, :], self.disp_scores[:, 1] = row2, col2
        # Add row/column 0 scores
        row3 = [char for _ in range(len2) for char in (0, " ")] + [0]
        col3 = [char for _ in range(len1) for char in (0, " ")] + [0]
        self.disp_scores[2, 2:], self.disp_scores[2:, 2] = row3, col3

    def score_linear(self):
        """
        Perform scoring with linear gap penalty. Fill self.scores and
        self.disp_scores.
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
                # Calculate and set final score
                final_score = max(score1, score2, score3, 0)
                self.scores[i, j] = final_score
                # Update self.disp_scores
                self.disp_scores[2*i+2, 2*j+2] = final_score
                if final_score == score1:
                    self.disp_scores[2 * i + 1, 2 * j + 1] = "↖"
                if final_score == score2:
                    self.disp_scores[2 * i + 1, 2 * j + 2] = "↑"
                if final_score == score3:
                    self.disp_scores[2 * i + 2, 2 * j + 1] = "←"
        self.high_score = np.max(self.scores)

    def traceback(self, i: int, j: int, align1: list, align2: list,
                  aligns: list[tuple[str, str]], curr_path=None):
        """
        Use recursion to perform traceback through the scoring matrix from a
        high score cell, finding all possible paths. Helper method for
        find_all_alignments().
        :param i: Row number of current cell
        :param j: Column number of current cell
        :param align1: Partial alignment of seq1 in current path
        :param align2: Partial alignment of seq2 in current path
        :param aligns: All optimal alignments of seq1 and seq2
        :param curr_path: Current partial alignment path
        """
        if curr_path is None:
            curr_path = list()

        # Base case: stop when score = 0 (first cell of alignment)
        if self.scores[i, j] == 0:
            # Record optimal alignment
            if align1 and align2:
                align1, align2 = reversed(align1), reversed(align2)
                aligns.append(("".join(align1), "".join(align2)))
            return

        # Mark cell as visited
        curr_path.append((i, j))

        # Explore all possible movement directions
        dirs = list()

        # Move diagonally ↖ (match)
        if self.disp_scores[2 * i + 1, 2 * j + 1] == "↖":
            dirs.append(("↖", i-1, j-1, align1 + [self.seq1[i-1]], align2 +
                         [self.seq2[j-1]]))

        # Move vertically ↑ (gap in seq2)
        if self.disp_scores[2 * i + 1, 2 * j + 2] == "↑":
            dirs.append(("↑", i-1, j, align1+[self.seq1[i-1]], align2 + ["-"]))

        # Move horizontally ← (gap in seq1)
        if self.disp_scores[2 * i + 2, 2 * j + 1] == "←":
            dirs.append(("←", i, j-1, align1+["-"], align2+[self.seq2[j-1]]))

        # Recurse in each direction
        for direction, next_i, next_j, next_align1, next_align2 in dirs:
            self.traceback(next_i, next_j, next_align1, next_align2, aligns,
                           curr_path[:])

        # Traceback - remove current cell from path
        curr_path.pop()

        return aligns

    def find_all_alignments(self):
        """
        Find all optimal alignments from the scoring matrix (by performing
        a traceback through the scoring matrix from all high scoring cells).
        Format alignments for printing.
        """
        # Find alignments
        all_aligns = list()
        # Traceback from all cells with max score
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.scores[i, j] == self.high_score:
                    self.traceback(i, j, [], [], all_aligns)
        self.alignments = all_aligns

        # Format alignments
        # Get lengths of all sequences in optimal alignments
        all_aligns = [seq for align in self.alignments for seq in align]
        seq_lens = [len(s) for s in all_aligns]
        # Generate matrix to display optimal alignments
        m_rows = len(all_aligns) + len(self.alignments) - 1
        m_cols = max(seq_lens)
        self.disp_alignments = \
            np.full((m_rows, m_cols), " ", dtype="<U1")
        # Assign each sequence to a row of the alignment display matrix
        row_nums = [i + (i // 2) for i in range(m_rows)]
        row_assign = list(zip(row_nums, all_aligns))
        for seq in row_assign:
            # Add whitespace to end of each sequence, then add to matrix
            self.disp_alignments[seq[0], :] = [_ for _ in seq[1].ljust(m_cols)]

    def sequence_alignment(self):
        """
        Populate scoring matrix and formats it for display. Find all optimal
        local alignments and format them for display.
        """
        # Error if sequence is empty string, whitespace, or None
        if (not self.seq1 or not self.seq2 or self.seq1.strip() == "" or
                self.seq2.strip() == ""):
            raise ValueError("Error: Empty sequence")

        self.score_linear()

        # Handle case of no optimal alignments
        if self.high_score == 0:
            return

        self.find_all_alignments()

    def __str__(self):
        """
        Return a string representation of the scoring matrix and alignments.
        """
        df_scores = pd.DataFrame(self.disp_scores)
        if self.alignments:
            df_align = pd.DataFrame(self.disp_alignments)
            str_rep = \
                (f"Scoring Matrix\n{df_scores.to_string(index=False, header=False)}"
                 + f"\n\nAlignments\n{df_align.to_string(index=False, header=False)}")
        else: # If no alignments, don't print disp_alignments
            str_rep = \
                (f"Scoring Matrix\n{df_scores.to_string(index=False, header=False)}"
                + f"\n\nAlignments\n{self.alignments}")
        return str_rep
