# coding: utf-8

"""
Stores distance metrics used in this work: Dynamic Time Warping (DTW) and Euclidean Distance.
It also stores the LB_Keough distance, which is used to compute a lower bound on DTW
"""

import numpy

__author__ = 'Joaquim Leitão'
__copyright__ = 'Copyright (c) 2017 Joaquim Leitão'
__email__ = 'jocaleitao93@gmail.com'


def dtw(s1, s2, w=None):
    # Calculates dynamic time warping Euclidean distance between two sequences. Option to enforce locality constraint
    # for window w.
    d_t_w = {}

    if w:
        w = max(w, abs(len(s1) - len(s2)))

        for i in range(-1, len(s1)):
            for j in range(-1, len(s2)):
                d_t_w[(i, j)] = float('inf')

    else:
        for i in range(len(s1)):
            d_t_w[(i, -1)] = float('inf')
        for i in range(len(s2)):
            d_t_w[(-1, i)] = float('inf')

    d_t_w[(-1, -1)] = 0

    for i in range(len(s1)):
        if w:
            for j in range(max(0, i - w), min(len(s2), i + w)):
                dist = (s1[i] - s2[j]) ** 2
                d_t_w[(i, j)] = dist + min(d_t_w[(i - 1, j)], d_t_w[(i, j - 1)], d_t_w[(i - 1, j - 1)])
        else:
            for j in range(len(s2)):
                dist = (s1[i] - s2[j]) ** 2
                d_t_w[(i, j)] = dist + min(d_t_w[(i - 1, j)], d_t_w[(i, j - 1)], d_t_w[(i - 1, j - 1)])

    return numpy.sqrt(d_t_w[len(s1) - 1, len(s2) - 1])


def lb_keogh(s1, s2, r):
    # Calculates LB_Keough lower bound to dynamic time warping. Linear complexity compared to quadratic complexity
    # of dtw
    l_b_sum = 0

    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            l_b_sum = l_b_sum + (i - upper_bound) ** 2
        elif i < lower_bound:
            l_b_sum = l_b_sum + (i - lower_bound) ** 2

    return numpy.sqrt(l_b_sum)
