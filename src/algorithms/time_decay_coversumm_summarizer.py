import math
import time
from datetime import datetime

import numpy as np
import graphgrove as gg

from copy import deepcopy
from graphgrove.sgtree import NNS_L2 as SGTree_NNS_L2
from algorithms.naivesgt_summarizer import SGTreeOnlineSummarizer


class TimeDecayCoverSummOnlineSummarizer(SGTreeOnlineSummarizer):
    def __init__(self, dim=100, min_capacity=100, alpha = 1e-1, summary_length=5, decay_rate=0.5, decay_type='exp'):
        # TODO: search what min_capacity and alpha used for
        super().__init__(summary_length=summary_length)
        self._current_neighbours = None
        self._current_neighbours_idx = None
        self._dim = dim
        self._max_dist = 1e6
        self._threshold = 0.
        self._capacity = 100
        self._min_capacity = min_capacity
        self._alpha = alpha
        self._last_mean = None
        self._decay_rate = decay_rate
        self._decay_type = decay_type

    def _get_decay_weight(self, date):
        date_diff_in_days = self._get_diff_date(date)

        if self._decay_type == 'power':
            return math.pow(date_diff_in_days, -self._decay_rate)
        elif self._decay_type == 'exp':
            return math.exp(-self._decay_rate * date_diff_in_days)
        elif self._decay_type == 'linear':
            return 1 / (1 + (self._decay_rate * date_diff_in_days))
        return math.exp(-self._decay_rate * date_diff_in_days)
        
    def _get_diff_date(self, review_date):
        current_date = datetime.now()
        date = datetime.fromisoformat(review_date)
        return (current_date - date).days

    def _return_knn(self, query):
        idx, dist = self._cover_tree.kNearestNeighbours(query.reshape(1, -1),
                                                        k=self._summary_length)
        self._max_dist = max(dist[0])
        self._current_neighbours_idx = idx[0]
        return self._current_neighbours_idx

    def _range_query(self, query, radius):
        idx, dist, neighbours = self._cover_tree.RangeSearch(
            query.reshape(1, -1), r=radius, return_points=True)
        self._current_neighbours = neighbours[0]
        self._current_neighbours_idx = idx[0]
        self._max_dist = max(dist[0])

    def _get_summary(self, query):
        distances = np.linalg.norm(self._current_neighbours - query, axis=-1)
        order = np.argsort(distances)[:self._summary_length]
        neighbour_idx = [self._current_neighbours_idx[o] for o in order]
        return neighbour_idx

    def update_summary(self, input_point, date):
        self._size += 1
        decay_weight = self._get_decay_weight(date)

        if self._size <= self._summary_length + 1:
            self._points.append(input_point)
        
        # update mean with time decay
        self._update_mean(input_point)
        self._current_mean = self._current_mean * decay_weight
        
        if self._size <= self._summary_length:
            return self._output_all()


        if self._cover_tree is None:
            self._cover_tree = SGTree_NNS_L2.from_matrix(np.array(
                self._points))
            self._last_mean = self._current_mean
            self._current_neighbours = deepcopy(self._points)
            self._current_neighbours_idx = list(range(self._size))
        else:
            self._cover_tree.insert(input_point[None, :])

        if self._size <= self._min_capacity:
            self._summary = self._return_knn(self._current_mean)
            self._last_mean = self._current_mean
            return self._summary
            
        if self._size > self._min_capacity:
            drift = np.linalg.norm(self._last_mean - self._current_mean)
            
            if drift >= self._threshold/2 or len(self._current_neighbours_idx) >= self._capacity:
                delta = self._min_capacity / self._size

                # compute threshold
                self._threshold = math.sqrt(
                    self._alpha * self._dim * math.log2(2 / delta) / 2 / self._size)
                
                # ct.ReservoirSearch
                self._summary = self._return_knn(self._current_mean)

                radius = self._threshold + self._max_dist
                self._range_query(self._current_mean, radius=radius)
                self._last_mean = self._current_mean
            else:
                if len(self._current_neighbours_idx) < self._capacity:
                    inp_dist = np.linalg.norm(self._last_mean - input_point)
                    if inp_dist < self._max_dist:
                        self._current_neighbours = np.append(self._current_neighbours, input_point[None, :], axis=0)
                        self._current_neighbours_idx = np.append(self._current_neighbours_idx, self._size - 1)
        
        self._summary = self._get_summary(self._current_mean)
        return self._summary
