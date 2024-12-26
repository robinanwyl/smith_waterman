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
        disp_scores: NumPy array of scoring matrix with directional arrows and
            row/column headings, formatted for display.
        high_score: Highest score in scoring matrix
        alignments: List of optimal local alignments
        disp_alignments: NumPy array of optimal local alignments, formatted
            for display.
    Methods:
        score_linear(): Calculates scoring matrix with linear gap penalty
        traceback(): Backtraces through the scoring matrix to find local
            alignments of seq1 and seq2, starting from a given high score cell
        find_all_alignments(): Finds all optimal alignments of seq1 and seq2
        sequence_alignment(): Calculates scoring matrix, finds all
            alignments, and formats scoring matrix and alignments for display
    """
    def __init__(self, seq1_, seq2_, match_, mismatch_, gap_penalty_):
        """
        Populates scoring matrix and determines optimal local alignments
        using Smith-Waterman algorithm with linear gap penalty. Formats
        scoring matrix and alignments for display.
        """
        self.seq1, self.seq2, = seq1_, seq2_
        self.match_score, self.mismatch_score, self.gap_penalty = \
            (match_, mismatch_, gap_penalty_)
        self.nrows, self.ncols = len(seq1_) + 1, len(seq2_) + 1
        self.scores = np.full((self.nrows, self.ncols), np.nan)
        self.disp_scores = np.full((2*self.nrows + 1, 2*self.ncols + 1),
                                    " ", dtype="<U5")
        self.high_score = 0
        self.alignments = list()
        self.disp_alignments = list()

        # Initial setup of self.scores
        self.scores[0, :] = 0 # Populate row 0 with scores of 0
        self.scores[:, 0] = 0 # Populate column 0 with scores of 0

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
        Generates scoring matrix with linear gap penalty. Populates self.scores
        and self.disp_scores.
        """
        gap_chars = ["-", "_"]
        for i in range(1, self.nrows):
            char1 = self.seq1[i-1]
            for j in range(1, self.ncols):
                char2 = self.seq2[j-1]
                # Calculate similarity score between characters
                if char1 == char2: # Match -> match score
                    sim_score = self.match_score
                else:
                    # Gap in either sequence -> gap penalty
                    if (char1 in gap_chars or char1.strip() == "" or char2 in
                            gap_chars or char2.strip() == ""):
                        sim_score = -1*self.gap_penalty
                    else: # Two mismatched nucleotides -> mismatch score
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
                self.disp_scores[2*i+2, 2*j+2] = int(final_score)
                # Add arrows to self.disp_scores
                if final_score == score1:
                    self.disp_scores[2 * i + 1, 2 * j + 1] = "↖"
                if final_score == score2:
                    self.disp_scores[2 * i + 1, 2 * j + 2] = "↑"
                if final_score == score3:
                    self.disp_scores[2 * i + 2, 2 * j + 1] = "←"
        # Cast self.scores to integer type
        self.scores = self.scores.astype(int)
        # Record high score
        self.high_score = np.max(self.scores)

    def traceback(self, i, j, align1, align2, aligns, curr_path=None,
                  all_paths=None):
        """
        Performs traceback through the scoring matrix from a high score cell.
        Helper method for find_all_alignments().
        :param i: (int) Row number of current cell
        :param j: (int) Column number of current cell
        :param align1: (list) Partial alignment of seq1 in current path
        :param align2: (list) Partial alignment of seq2 in current path
        :param aligns: (list of tuples) All optimal alignments of seq1 and seq2
        :param curr_path: (list) Current partial alignment path
        :param all_paths: (list) All optimal alignment paths
        """
        if curr_path is None:
            curr_path = list()
        if all_paths is None:
            all_paths = list()

        # Base case: stop when score = 0 (first cell of alignment)
        if self.scores[i, j] == 0:
            # Record current path
            all_paths.append(curr_path[:])
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
                           curr_path[:], all_paths)

        # Traceback - remove current cell from path
        curr_path.pop()

        return aligns

    def find_all_alignments(self):
        """
        Finds all optimal local alignments from the scoring matrix -
        backtracks through the scoring matrix from all high scoring cells.
        Formats alignments for printing.
        """
        # Find alignments
        all_aligns = list()
        # Traceback from all cells with max score
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.scores[i, j] == self.high_score:
                    self.traceback(i, j, [], [], all_aligns)
        self.alignments = all_aligns
        print(f"self.alignments = {self.alignments}")

        # Format alignments
        # Get lengths of all sequences in optimal alignments
        all_aligns = [seq for align in self.alignments for seq in align]
        print(f"all_aligns = {all_aligns}")
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
        Populates scoring matrix and formats it for display. Finds all optimal
        local alignments and formats them for display.
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
        df_scores = pd.DataFrame(self.disp_scores)
        if self.alignments:
            df_align = pd.DataFrame(self.disp_alignments)
            to_print = \
                (f"Scoring Matrix\n{df_scores.to_string(index=False, header=False)}"
                 + f"\n\nAlignments\n{df_align.to_string(index=False, header=False)}")
        else: # If no alignments, don't print disp_alignments
            to_print = \
                (f"Scoring Matrix\n{df_scores.to_string(index=False, header=False)}"
                + f"\n\nAlignments\n{self.alignments}")
        return to_print


# Testing code
matrix = ScoringMatrix("AC-GT", "ACG-T", 2, -1, 3)
matrix.sequence_alignment()
print(matrix.high_score)
print(matrix.alignments)
print(matrix)