import json
import sqlite3
import traceback
from __future__ import print_function
import os, sys

import argparse


class initializor:
    def __init__(self):
        self.partial_scores = None

    def loadw_HDS(self, sql):
        get_comp1_ = get_component1(sql)
        get_comp2_ = get_component2(sql)
        get_others_ = get_others(sql)

        if get_comp1_ <= 1 and get_others_ == 0 and get_comp2_ == 0:
            return "easy"
        elif (get_others_ <= 2 and get_comp1_ <= 1 and get_comp2_ == 0) or \
                (get_comp1_ <= 2 and get_others_ < 2 and get_comp2_ == 0):
            return "medium"
        elif (get_others_ > 2 and get_comp1_ <= 2 and get_comp2_ == 0) or \
                (2 < get_comp1_ <= 3 and get_others_ <= 2 and get_comp2_ == 0) or \
                (get_comp1_ <= 1 and get_others_ == 0 and get_comp2_ <= 1):
            return "hard"
        else:
            return "extra"

    def loadw_exact_match(self, pred, label):
        partial_scores = self.loadw_partial_match(pred, label)
        self.partial_scores = partial_scores

        for _, score in partial_scores.items():
            if score['ST1'] != 1:
                return 0
        if len(label['from']['table_units']) > 0:
            label_tables = sorted(label['from']['table_units'])
            pred_tables = sorted(pred['from']['table_units'])
            return label_tables == pred_tables
        return 1

    def loadw_partial_match(self, pred, label):
        res = {}

        label_total, pred_total, cnt, cnt_wo_agg = loadv_p(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['select'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, ST1 = loads_scores(cnt_wo_agg, pred_total, label_total)
        res['select(no AGG)'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt, cnt_wo_agg = loadw_where(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['where'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}
        acc, rec, ST1 = loads_scores(cnt_wo_agg, pred_total, label_total)
        res['where(no OP)'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_group(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['group(no Having)'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_having(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['group'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_order(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['order'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_and_or(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['and/or'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_IUEN(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['IUEN'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        label_total, pred_total, cnt = loadw_keywords(pred, label)
        acc, rec, ST1 = loads_scores(cnt, pred_total, label_total)
        res['keywords'] = {'acc': acc, 'rec': rec, 'ST1': ST1,'label_total':label_total,'pred_total':pred_total}

        return res
