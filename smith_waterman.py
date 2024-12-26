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
        alignments: List of optimal local alignments
        disp_alignments: NumPy array of optimal local alignments, formatted
            for display.
    Methods:
        score_linear(): Calculates scoring matrix with linear gap penalty
        format_scoring_matrix(): Formats the scoring matrix with arrows and
            row/column headings
        traceback(): Backtraces through the scoring matrix to find local
            alignments of seq1 and seq2, starting from a given high score cell
        find_all_alignments(): Finds all optimal alignments of seq1 and seq2
        format_alignments(): Formats optimal alignments for display
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
        self.alignments = list()
        self.disp_alignments = None

        # Initial setup of self.scores
        self.scores[0, :] = 0 # Populate row 0 with scores of 0
        self.scores[:, 0] = 0 # Populate column 0 with scores of 0

        # Scoring and display
        self.score_linear() # Calculate scores
        self.format_scores() # Populate formatted scoring matrix
        self.find_all_alignments() # Determine local optimal alignments
        self.format_alignments()  # Format alignments for display

    def score_linear(self):
        """
        Generates scoring matrix with linear gap penalty. Populates self.scores
        and self.disp_scores.
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

    def format_scores(self):
        """
        Generates formatted scoring matrix with directional arrows and
        row/column headings.
        """
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
        # Add rest of scoring matrix and arrows
        self.disp_scores[4::2, 4::2] = self.scores[1:, 1:]
        for i in range(1, self.nrows):
            for j in range(1, self.ncols):
                if (self.scores[i, j] == self.scores[i-1, j-1] + self.match_score or
                        self.scores[i, j] == self.scores[i-1, j-1] + self.mismatch_score):
                    self.disp_scores[2 * i + 1, 2 * j + 1] = "↖"
                if self.scores[i, j] == self.scores[
                    i - 1, j] - self.gap_penalty:
                    self.disp_scores[2 * i + 1, 2 * j + 2] = "↑"
                if self.scores[i, j] == self.scores[
                    i, j - 1] - self.gap_penalty:
                    self.disp_scores[2 * i + 2, 2 * j + 1] = "←"

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
        print(f"current (i, j): ({i}, {j}), score = {self.scores[i, j]}")
        if curr_path is None:
            curr_path = list()
        if all_paths is None:
            all_paths = list()

        # Base case: stop when score = 0 (first cell of alignment)
        if self.scores[i, j] == 0:
            print("base case")
            # Record current path
            all_paths.append(curr_path[:])
            # Check if current path has high score
            print(f"curr_path = {curr_path}")
            diffs = sum([self.scores[curr_path[n][0], curr_path[n][1]] -
                         self.scores[curr_path[n+1][0], curr_path[n+1][1]]
                         for n in range(len(curr_path)-1)]
                        + [self.scores[curr_path[-1][0], curr_path[-1][1]]])
            print(f"diffs {diffs}")
            if diffs == self.scores[curr_path[0][0], curr_path[0][1]]:
                print(f"high score {self.scores[curr_path[0][0], curr_path[0][1]]}")
                # Record optimal alignment
                align1, align2 = reversed(align1), reversed(align2)
                aligns.append(("".join(align1), "".join(align2)))
                print(f"aligns {aligns}")
            return

        # Mark cell as visited
        curr_path.append((i, j))

        # Explore all possible movement directions
        dirs = list()

        # Move diagonally ↖ (match)
        if self.disp_scores[2 * i + 1, 2 * j + 1] == "↖":
            print(f"align1 {align1}, align2 {align2}")
            print(f"aligns {aligns}")
            print("↖")
            print(f"next (i, j) {(i-1, j-1)}")
            dirs.append(("↖", i-1, j-1, align1 + [self.seq1[i-1]], align2 +
                         [self.seq2[j-1]]))

        # Move vertically ↑ (gap in seq2)
        if self.disp_scores[2 * i + 1, 2 * j + 2] == "↑":
            print(f"align1 {align1}, align2 {align2}")
            print(f"aligns {aligns}")
            print("↑")
            print(f"next (i, j) {(i-1, j)}")
            dirs.append(("↑", i-1, j, align1+[self.seq1[i-1]], align2 + ["-"]))

        # Move horizontally ← (gap in seq1)
        if self.disp_scores[2 * i + 2, 2 * j + 1] == "←":
            print(f"align1 {align1}, align2 {align2}")
            print(f"aligns {aligns}")
            print("←")
            print(f"next (i, j) {(i, j-1)}")
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
        Finds all optimal local alignments from the scoring matrix. Backtraces
        through the scoring matrix from all high scoring cells.
        """
        all_aligns = list()
        max_score = np.max(self.scores)
        # Traceback from all cells with max score
        for i in range(self.nrows):
            for j in range(self.ncols):
                if self.scores[i, j] == max_score:
                    print("find_all_alignments(): max score found")
                    self.traceback(i, j, [], [], all_aligns)
        self.alignments = all_aligns
        print(f"all alignments: {self.alignments}")

    def format_alignments(self):
        """
        Populates self.disp_alignments NumPy array with the optimal local
        alignments of seq1 and seq2, for display.
        """
        pass
        # Get lengths of all sequences in optimal alignments
        all_aligns = [seq for align in self.alignments for seq in align]
        seq_lens = [len(s) for s in all_aligns]
        # Generate matrix to display optimal alignments
        m_rows = len(all_aligns)+len(self.alignments)-1
        m_cols = max(seq_lens)
        self.disp_alignments = \
            np.full((m_rows, m_cols)," ", dtype="<U1")
        # Assign each sequence to a row of the alignment display matrix
        row_nums = [i + (i//2) for i in range(m_rows)]
        row_assign = list(zip(row_nums, all_aligns))
        for seq in row_assign:
            # Add whitespace to end of each sequence, then add to matrix
            self.disp_alignments[seq[0], :] = [_ for _ in seq[1].ljust(m_cols)]

    def __str__(self):
        df_scores = pd.DataFrame(self.disp_scores)
        df_align = pd.DataFrame(self.disp_alignments)
        to_print = f"{df_scores.to_string(index=False, header=False)}" + \
                    "\n\n" + f"{df_align.to_string(index=False, header=False)}"
        return to_print


# Testing code
matrix = ScoringMatrix("GGATCGA", "GAATTCAGTTA", 5, -3, 4)
print(matrix)