# -*- coding: utf-8 -*-

from .aggregator import SumAggregator, ConcatAggregator, NeighAggregator,featureAggregator, AvgAggregator

Aggregator = {
    'sum': SumAggregator,
    'concat': ConcatAggregator,
    'neigh': NeighAggregator,
    'feature':featureAggregator,
    'average': AvgAggregator
}
