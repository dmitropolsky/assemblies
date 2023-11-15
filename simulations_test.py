#! /usr/bin/python

import brain_util as bu
import simulations
import unittest

class TestBrainFunction(unittest.TestCase):
    def test_projection(self):
        w = simulations.project_sim(1000000, 1000, 0.001, 0.05, 25)
        self.assertEqual(w[-2], w[-1])

    def test_pattern_completion(self):
        (_, winners) = simulations.pattern_com(
            100000, 317, 0.05, 0.05, 25, 0.5, 5)
        self.assertGreaterEqual(bu.overlap(winners[24], winners[29]), 300)

    def test_association(self):
        (_, winners) = simulations.association_sim(100000, 317, 0.05, 0.1, 10)
        self.assertLessEqual(bu.overlap(winners[9], winners[19]), 2)
        self.assertGreaterEqual(bu.overlap(winners[9], winners[29]), 100)
        self.assertGreaterEqual(bu.overlap(winners[19], winners[29]), 100)
        self.assertGreaterEqual(bu.overlap(winners[9], winners[39]), 20)

    def test_merge(self):
        (w_a, w_b, w_c) = simulations.merge_sim(100000,317,0.01,0.05,50)
        self.assertLessEqual(w_a[-1], 3200)
        self.assertLessEqual(w_b[-1], 3200)
        self.assertLessEqual(w_c[-1], 6400)


if __name__ == '__main__':
    unittest.main()
