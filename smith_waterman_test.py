"""
Smith-Waterman Algorithm Testing
Robin Anwyl
December 2024
Description: Testing Smith-Waterman algorithm implementation.
"""

import unittest
from smith_waterman import *

class SmithWatermanTests(unittest.TestCase):
    # Simple tests
    def testCompleteMatch(self):
        seq1, seq2 = "ACGT", "ACGT"
        actual = ScoringMatrix(seq1, seq2, 2, -1, 3)
        actual.sequence_alignment()
        actual_score, actual_aligns = actual.high_score, actual.alignments
        expected_score, expected_aligns = 8, [("ACGT", "ACGT")]
        self.assertEqual(expected_score, actual_score)
        self.assertEqual(expected_aligns, actual_aligns)

    def testCompleteMismatch(self):
        seq1, seq2 = "AACC", "GGTT"
        actual = ScoringMatrix(seq1, seq2, 2, -1, 3)
        actual.sequence_alignment()
        actual_score, actual_aligns = actual.high_score, actual.alignments
        expected_score, expected_aligns = 0, []
        self.assertEqual(expected_score, actual_score)
        self.assertEqual(expected_aligns, actual_aligns)

    def testDifferentLengths(self):
        seq1, seq2 = "A", "ACGT"
        actual = ScoringMatrix(seq1, seq2, 2, -1, 3)
        actual.sequence_alignment()
        actual_score, actual_aligns = actual.high_score, actual.alignments
        expected_score, expected_aligns = 2, [("A", "A")]
        self.assertEqual(expected_score, actual_score)
        self.assertEqual(expected_aligns, actual_aligns)

    def testEmptyString(self):
        seq1, seq2 = "", "ACGT"
        actual = ScoringMatrix(seq1, seq2, 2, -1, 3)
        with self.assertRaises(ValueError):
            actual.sequence_alignment()

    def testWhitespace(self):
        seq1, seq2 = "   ", "ACGT"
        actual = ScoringMatrix(seq1, seq2, 2, -1, 3)
        with self.assertRaises(ValueError):
            actual.sequence_alignment()

    def testGaps(self):
        pass

    def testSequences1(self): # Sequences from BioGem.org
        seq1, seq2 = "GGATCGA", "GAATTCAGTTA"
        actual = ScoringMatrix(seq1, seq2, 5, -3, 4)
        actual.sequence_alignment()
        actual_score, actual_aligns = actual.high_score, actual.alignments
        expected_score = 14
        expected_aligns = [('GGA-TC-G', 'GAATTCAG'), ('GGAT-C-G', 'GAATTCAG'),
                           ('GGA-TCGA', 'GAATTC-A'), ('GGAT-CGA', 'GAATTC-A')]
        self.assertEqual(expected_score, actual_score)
        self.assertEqual(set(expected_aligns), set(actual_aligns))

    def testSequences2(self): # Sequences from Wikipedia
        seq1, seq2 = "GGTTGACTA", "TGTTACGG"
        actual = ScoringMatrix(seq1, seq2, 3, -3, 2)
        actual.sequence_alignment()
        actual_score, actual_aligns = actual.high_score, actual.alignments
        expected_score, expected_aligns = 13, [('GTTGAC', 'GTT-AC')]
        self.assertEqual(expected_score, actual_score)
        self.assertEqual(set(expected_aligns), set(actual_aligns))