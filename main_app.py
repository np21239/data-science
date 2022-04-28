from m_constants import *
from __future__ import print_function
import os, sys
import json
import sqlite3
import traceback
from initializor import *
import argparse

from engine import loadData, loads_mapping, loads_alias_words, mapping, loadSQL


def statet_has_or(conds):
    return 'or' in conds[1::2]


def statet_has_like(conds):
    return OPERATIONAL_WHERE.index('like') in [cond_unit[1] for cond_unit in conds[::2]]


def statet_has_sql(conds):
    for cond_unit in conds[::2]:
        val1, val2 = cond_unit[3], cond_unit[4]
        if val1 is not None and type(val1) is dict:
            return True
        if val2 is not None and type(val2) is dict:
            return True
    return False


def containsops(val_unit):
    return val_unit[0] != SIOPS.index('none')


def containsags(unit):
    return unit[0] != ADDOPS.index('none')


def tespps(count, total):
    if count == total:
        return 1
    return 0


def resls(count, total):
    if count == total:
        return 1
    return 0


def ST1(acc, rec):
    if (acc + rec) == 0:
        return 0
    return (2. * acc * rec) / (acc + rec)


def loads_scores(count, predict_total, label_total):
    if predict_total != label_total:
        return 0,0,0
    elif count == predict_total:
        return 1,1,1
    return 0,0,0


def loadv_p(predict, label):
    predict_sel = predict['select'][1]
    label_sel = label['select'][1]
    label_wo_agg = [unit[1] for unit in label_sel]
    predict_total = len(predict_sel)
    label_total = len(label_sel)
    cnt = 0
    cnt_wo_agg = 0

    for unit in predict_sel:
        if unit in label_sel:
            cnt += 1
            label_sel.remove(unit)
        if unit[1] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[1])

    return label_total, predict_total, cnt, cnt_wo_agg


def loadw_where(predict, label):
    predict_conds = [unit for unit in predict['where'][::2]]
    label_conds = [unit for unit in label['where'][::2]]
    label_wo_agg = [unit[2] for unit in label_conds]
    predict_total = len(predict_conds)
    label_total = len(label_conds)
    cnt = 0
    cnt_wo_agg = 0

    for unit in predict_conds:
        if unit in label_conds:
            cnt += 1
            label_conds.remove(unit)
        if unit[2] in label_wo_agg:
            cnt_wo_agg += 1
            label_wo_agg.remove(unit[2])

    return label_total, predict_total, cnt, cnt_wo_agg


def loadw_group(predict, label):
    predict_cols = [unit[1] for unit in predict['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    predict_total = len(predict_cols)
    label_total = len(label_cols)
    cnt = 0
    predict_cols = [predict.split(".")[1] if "." in predict else predict for predict in predict_cols]
    label_cols = [label.split(".")[1] if "." in label else label for label in label_cols]
    for col in predict_cols:
        if col in label_cols:
            cnt += 1
            label_cols.remove(col)
    return label_total, predict_total, cnt


def loadw_having(predict, label):
    predict_total = label_total = cnt = 0
    if len(predict['groupBy']) > 0:
        predict_total = 1
    if len(label['groupBy']) > 0:
        label_total = 1

    predict_cols = [unit[1] for unit in predict['groupBy']]
    label_cols = [unit[1] for unit in label['groupBy']]
    if predict_total == label_total == 1 \
            and predict_cols == label_cols \
            and predict['having'] == label['having']:
        cnt = 1

    return label_total, predict_total, cnt


def loadw_order(predict, label):
    predict_total = label_total = cnt = 0
    if len(predict['orderBy']) > 0:
        predict_total = 1
    if len(label['orderBy']) > 0:
        label_total = 1
    if len(label['orderBy']) > 0 and predict['orderBy'] == label['orderBy'] and \
            ((predict['limit'] is None and label['limit'] is None) or (predict['limit'] is not None and label['limit'] is not None)):
        cnt = 1
    return label_total, predict_total, cnt


def loadw_and_or(predict, label):
    predict_ao = predict['where'][1::2]
    label_ao = label['where'][1::2]
    predict_ao = set(predict_ao)
    label_ao = set(label_ao)

    if predict_ao == label_ao:
        return 1,1,1
    return len(predict_ao),len(label_ao),0


def loads_nestedSQL(sql):
    nested = []
    for cond_unit in sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]:
        if type(cond_unit[3]) is dict:
            nested.append(cond_unit[3])
        if type(cond_unit[4]) is dict:
            nested.append(cond_unit[4])
    if sql['intersect'] is not None:
        nested.append(sql['intersect'])
    if sql['except'] is not None:
        nested.append(sql['except'])
    if sql['union'] is not None:
        nested.append(sql['union'])
    return nested


def loadw_nested(predict, label):
    label_total = 0
    predict_total = 0
    cnt = 0
    if predict is not None:
        predict_total += 1
    if label is not None:
        label_total += 1
    if predict is not None and label is not None:
        cnt += initializor().loadw_exact_match(predict, label)
    return label_total, predict_total, cnt


def loadw_IUEN(predict, label):
    lt1, pt1, cnt1 = loadw_nested(predict['intersect'], label['intersect'])
    lt2, pt2, cnt2 = loadw_nested(predict['except'], label['except'])
    lt3, pt3, cnt3 = loadw_nested(predict['union'], label['union'])
    label_total = lt1 + lt2 + lt3
    predict_total = pt1 + pt2 + pt3
    cnt = cnt1 + cnt2 + cnt3
    return label_total, predict_total, cnt


def loads_keywords(sql):
    res = set()
    if len(sql['where']) > 0:
        res.add('where')
    if len(sql['groupBy']) > 0:
        res.add('group')
    if len(sql['having']) > 0:
        res.add('having')
    if len(sql['orderBy']) > 0:
        res.add(sql['orderBy'][0])
        res.add('order')
    if sql['limit'] is not None:
        res.add('limit')
    if sql['except'] is not None:
        res.add('except')
    if sql['union'] is not None:
        res.add('union')
    if sql['intersect'] is not None:
        res.add('intersect')
    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    if len([token for token in ao if token == 'or']) > 0:
        res.add('or')
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    if len([cond_unit for cond_unit in cond_units if cond_unit[0]]) > 0:
        res.add('not')
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == OPERATIONAL_WHERE.index('in')]) > 0:
        res.add('in')
    if len([cond_unit for cond_unit in cond_units if cond_unit[1] == OPERATIONAL_WHERE.index('like')]) > 0:
        res.add('like')
    return res
def loadw_keywords(predict, label):
    predict_keywords = loads_keywords(predict)
    label_keywords = loads_keywords(label)
    predict_total = len(predict_keywords)
    label_total = len(label_keywords)
    cnt = 0
    for k in predict_keywords:
        if k in label_keywords:
            cnt += 1
    return label_total, predict_total, cnt
def get_agg(units):
    return len([unit for unit in units if containsags(unit)])
def get_component1(sql):
    count = 0
    if len(sql['where']) > 0:
        count += 1
    if len(sql['groupBy']) > 0:
        count += 1
    if len(sql['orderBy']) > 0:
        count += 1
    if sql['limit'] is not None:
        count += 1
    if len(sql['from']['table_units']) > 0:  # JOIN
        count += len(sql['from']['table_units']) - 1

    ao = sql['from']['conds'][1::2] + sql['where'][1::2] + sql['having'][1::2]
    count += len([token for token in ao if token == 'or'])
    cond_units = sql['from']['conds'][::2] + sql['where'][::2] + sql['having'][::2]
    count += len([cond_unit for cond_unit in cond_units if cond_unit[1] == OPERATIONAL_WHERE.index('like')])

    return count


def get_component2(sql):
    nested = loads_nestedSQL(sql)
    return len(nested)

def get_others(sql):
    count = 0
    agg_count = get_agg(sql['select'][1])
    agg_count += get_agg(sql['where'][::2])
    agg_count += get_agg(sql['groupBy'])
    if len(sql['orderBy']) > 0:
        agg_count += get_agg([unit[1] for unit in sql['orderBy'][1] if unit[1]] +
                            [unit[2] for unit in sql['orderBy'][1] if unit[2]])
    agg_count += get_agg(sql['having'])
    if agg_count > 1:
        count += 1
    if len(sql['select'][1]) > 1:
        count += 1
    if len(sql['where']) > 1:
        count += 1
    if len(sql['groupBy']) > 1:
        count += 1
    return count





def isValidSQL(sql, db):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
    except:
        return False
    return True


def print_scores(scores, etype):
    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']

    print("{:20} {:20} {:20} {:20} {:20} {:20}".format("", *levels))
    counts = [scores[level]['count'] for level in levels]
    print("{:20} {:<20d} {:<20d} {:<20d} {:<20d} {:<20d}".format("count", *counts))

    if etype in ["all", "exec"]:
        this_scores = [scores[level]['exec'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("execution", *this_scores))

    if etype in ["all", "match"]:
        exact_scores = [scores[level]['exact'] for level in levels]
        print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format("exact match", *exact_scores))
      
        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['acc'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['rec'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))

        for type_ in partial_types:
            this_scores = [scores[level]['partial'][type_]['ST1'] for level in levels]
            print("{:20} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f} {:<20.3f}".format(type_, *this_scores))


def evaluate(gldmain, predictict, db_dir, etype, kmaps):
    with open(gldmain) as f:
        glist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]

    with open(predictict) as f:
        plist = [l.strip().split('\t') for l in f.readlines() if len(l.strip()) > 0]
        initializor = initializor()

    levels = ['easy', 'medium', 'hard', 'extra', 'all']
    partial_types = ['select', 'select(no AGG)', 'where', 'where(no OP)', 'group(no Having)',
                     'group', 'order', 'and/or', 'IUEN', 'keywords']
    entries = []
    scores = {}

    for level in levels:
        scores[level] = {'count': 0, 'partial': {}, 'exact': 0.}
        scores[level]['exec'] = 0
        for type_ in partial_types:
            scores[level]['partial'][type_] = {'acc': 0., 'rec': 0., 'ST1': 0.,'acc_count':0,'rec_count':0}

    loadw_err_num = 0
    for p, g in zip(plist, glist):
        p_str = p[0]
        g_str, db = g
        db_name = db
        db = os.path.join(db_dir, db, db + ".sqlite")
        mapping = mapping(loads_mapping(db))
        g_sql = loadSQL(mapping, g_str)
        HDS = initializor.loadw_HDS(g_sql)
        scores[HDS]['count'] += 1
        scores['all']['count'] += 1

        try:
            p_sql = loadSQL(mapping, p_str)
        except:
            # If p_sql is not valid, then we will use an empty sql to evaluate with the correct sql
            p_sql = {
            "except": None,
            "from": {
                "conds": [],
                "table_units": []
            },
            "groupBy": [],
            "having": [],
            "intersect": None,
            "limit": None,
            "orderBy": [],
            "select": [
                False,
                []
            ],
            "union": None,
            "where": []
            }
            loadw_err_num += 1
            print("loadw_err_num:{}".format(loadw_err_num))

        # rebuild sql for value evaluation
        kmap = kmaps[db_name]
        g_valid_col_units = build_valid_col_units(g_sql['from']['table_units'], mapping)
        g_sql = rebuild_sql_val(g_sql)
        g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
        p_valid_col_units = build_valid_col_units(p_sql['from']['table_units'], mapping)
        p_sql = rebuild_sql_val(p_sql)
        p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

        if etype in ["all", "exec"]:
            exec_score = loadw_exec_match(db, p_str, g_str, p_sql, g_sql)
            if exec_score:
                scores[HDS]['exec'] += 1.0
                scores['all']['exec'] += 1.0

        if etype in ["all", "match"]:
            exact_score = initializor.loadw_exact_match(p_sql, g_sql)
            partial_scores = initializor.partial_scores
            if exact_score == 0:
                print("{} predict: {}".format(HDS,p_str))
                print("{} gldmain: {}".format(HDS,g_str))
                print("")
            scores[HDS]['exact'] += exact_score
            scores['all']['exact'] += exact_score
            for type_ in partial_types:
                if partial_scores[type_]['predict_total'] > 0:
                    scores[HDS]['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores[HDS]['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores[HDS]['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores[HDS]['partial'][type_]['rec_count'] += 1
                scores[HDS]['partial'][type_]['ST1'] += partial_scores[type_]['ST1']
                if partial_scores[type_]['predict_total'] > 0:
                    scores['all']['partial'][type_]['acc'] += partial_scores[type_]['acc']
                    scores['all']['partial'][type_]['acc_count'] += 1
                if partial_scores[type_]['label_total'] > 0:
                    scores['all']['partial'][type_]['rec'] += partial_scores[type_]['rec']
                    scores['all']['partial'][type_]['rec_count'] += 1
                scores['all']['partial'][type_]['ST1'] += partial_scores[type_]['ST1']

            entries.append({
                'predictictSQL': p_str,
                'gldmainSQL': g_str,
                'HDS': HDS,
                'exact': exact_score,
                'partial': partial_scores
            })

    for level in levels:
        if scores[level]['count'] == 0:
            continue
        if etype in ["all", "exec"]:
            scores[level]['exec'] /= scores[level]['count']

        if etype in ["all", "match"]:
            scores[level]['exact'] /= scores[level]['count']
            for type_ in partial_types:
                if scores[level]['partial'][type_]['acc_count'] == 0:
                    scores[level]['partial'][type_]['acc'] = 0
                else:
                    scores[level]['partial'][type_]['acc'] = scores[level]['partial'][type_]['acc'] / \
                                                             scores[level]['partial'][type_]['acc_count'] * 1.0
                if scores[level]['partial'][type_]['rec_count'] == 0:
                    scores[level]['partial'][type_]['rec'] = 0
                else:
                    scores[level]['partial'][type_]['rec'] = scores[level]['partial'][type_]['rec'] / \
                                                             scores[level]['partial'][type_]['rec_count'] * 1.0
                if scores[level]['partial'][type_]['acc'] == 0 and scores[level]['partial'][type_]['rec'] == 0:
                    scores[level]['partial'][type_]['ST1'] = 1
                else:
                    scores[level]['partial'][type_]['ST1'] = \
                        2.0 * scores[level]['partial'][type_]['acc'] * scores[level]['partial'][type_]['rec'] / (
                        scores[level]['partial'][type_]['rec'] + scores[level]['partial'][type_]['acc'])

    print_scores(scores, etype)


def loadw_exec_match(db, p_str, g_str, predict, gldmain):
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    try:
        cursor.execute(p_str)
        p_res = cursor.fetchall()
    except:
        return False

    cursor.execute(g_str)
    q_res = cursor.fetchall()

    def res_map(res, val_units):
        rmap = {}
        for idx, val_unit in enumerate(val_units):
            key = tuple(val_unit[1]) if not val_unit[2] else (val_unit[0], tuple(val_unit[1]), tuple(val_unit[2]))
            rmap[key] = [r[idx] for r in res]
        return rmap

    p_val_units = [unit[1] for unit in predict['select'][1]]
    q_val_units = [unit[1] for unit in gldmain['select'][1]]
    return res_map(p_res, p_val_units) == res_map(q_res, q_val_units)

def rebuild_cond_unit_val(cond_unit):
    if cond_unit is None or not REM_VAL:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    if type(val1) is not dict:
        val1 = None
    else:
        val1 = rebuild_sql_val(val1)
    if type(val2) is not dict:
        val2 = None
    else:
        val2 = rebuild_sql_val(val2)
    return not_op, op_id, val_unit, val1, val2


def rebuild_statet_val(statet):
    if statet is None or not REM_VAL:
        return statet

    res = []
    for idx, it in enumerate(statet):
        if idx % 2 == 0:
            res.append(rebuild_cond_unit_val(it))
        else:
            res.append(it)
    return res


def rebuild_sql_val(sql):
    if sql is None or not REM_VAL:
        return sql

    sql['from']['conds'] = rebuild_statet_val(sql['from']['conds'])
    sql['having'] = rebuild_statet_val(sql['having'])
    sql['where'] = rebuild_statet_val(sql['where'])
    sql['intersect'] = rebuild_sql_val(sql['intersect'])
    sql['except'] = rebuild_sql_val(sql['except'])
    sql['union'] = rebuild_sql_val(sql['union'])

    return sql

def build_valid_col_units(table_units, mapping):
    col_ids = [table_unit[1] for table_unit in table_units if table_unit[0] == TBTYP['table_unit']]
    prefixs = [col_id[:-2] for col_id in col_ids]
    valid_col_units= []
    for value in mapping.idMap.values():
        if '.' in value and value[:value.index('.')] in prefixs:
            valid_col_units.append(value)
    return valid_col_units


def rebuild_col_unit_col(valid_col_units, col_unit, kmap):
    if col_unit is None:
        return col_unit

    agg_id, col_id, distinct = col_unit
    if col_id in kmap and col_id in valid_col_units:
        col_id = kmap[col_id]
    if REM_DISTN:
        distinct = None
    return agg_id, col_id, distinct


def rebuild_val_unit_col(valid_col_units, val_unit, kmap):
    if val_unit is None:
        return val_unit

    unit_op, col_unit1, col_unit2 = val_unit
    col_unit1 = rebuild_col_unit_col(valid_col_units, col_unit1, kmap)
    col_unit2 = rebuild_col_unit_col(valid_col_units, col_unit2, kmap)
    return unit_op, col_unit1, col_unit2


def rebuild_table_unit_col(valid_col_units, table_unit, kmap):
    if table_unit is None:
        return table_unit

    TBTYP, col_unit_or_sql = table_unit
    if isinstance(col_unit_or_sql, tuple):
        col_unit_or_sql = rebuild_col_unit_col(valid_col_units, col_unit_or_sql, kmap)
    return TBTYP, col_unit_or_sql


def rebuild_cond_unit_col(valid_col_units, cond_unit, kmap):
    if cond_unit is None:
        return cond_unit

    not_op, op_id, val_unit, val1, val2 = cond_unit
    val_unit = rebuild_val_unit_col(valid_col_units, val_unit, kmap)
    return not_op, op_id, val_unit, val1, val2


def rebuild_statet_col(valid_col_units, statet, kmap):
    for idx in range(len(statet)):
        if idx % 2 == 0:
            statet[idx] = rebuild_cond_unit_col(valid_col_units, statet[idx], kmap)
    return statet


def rebuild_select_col(valid_col_units, sel, kmap):
    if sel is None:
        return sel
    distinct, _list = sel
    new_list = []
    for it in _list:
        agg_id, val_unit = it
        new_list.append((agg_id, rebuild_val_unit_col(valid_col_units, val_unit, kmap)))
    if REM_DISTN:
        distinct = None
    return distinct, new_list


def rebuild_from_col(valid_col_units, from_, kmap):
    if from_ is None:
        return from_

    from_['table_units'] = [rebuild_table_unit_col(valid_col_units, table_unit, kmap) for table_unit in from_['table_units']]
    from_['conds'] = rebuild_statet_col(valid_col_units, from_['conds'], kmap)
    return from_


def rebuild_group_by_col(valid_col_units, group_by, kmap):
    if group_by is None:
        return group_by

    return [rebuild_col_unit_col(valid_col_units, col_unit, kmap) for col_unit in group_by]


def rebuild_order_by_col(valid_col_units, order_by, kmap):
    if order_by is None or len(order_by) == 0:
        return order_by

    direction, val_units = order_by
    new_val_units = [rebuild_val_unit_col(valid_col_units, val_unit, kmap) for val_unit in val_units]
    return direction, new_val_units


def rebuild_sql_col(valid_col_units, sql, kmap):
    if sql is None:
        return sql

    sql['select'] = rebuild_select_col(valid_col_units, sql['select'], kmap)
    sql['from'] = rebuild_from_col(valid_col_units, sql['from'], kmap)
    sql['where'] = rebuild_statet_col(valid_col_units, sql['where'], kmap)
    sql['groupBy'] = rebuild_group_by_col(valid_col_units, sql['groupBy'], kmap)
    sql['orderBy'] = rebuild_order_by_col(valid_col_units, sql['orderBy'], kmap)
    sql['having'] = rebuild_statet_col(valid_col_units, sql['having'], kmap)
    sql['intersect'] = rebuild_sql_col(valid_col_units, sql['intersect'], kmap)
    sql['except'] = rebuild_sql_col(valid_col_units, sql['except'], kmap)
    sql['union'] = rebuild_sql_col(valid_col_units, sql['union'], kmap)

    return sql


def build_foreign_key_map(entry):
    cols_orig = entry["column_names_original"]
    tables_orig = entry["table_names_original"]
    cols = []
    for col_orig in cols_orig:
        if col_orig[0] >= 0:
            t = tables_orig[col_orig[0]]
            c = col_orig[1]
            cols.append("__" + t.lower() + "." + c.lower() + "__")
        else:
            cols.append("__all__")

    def keyset_in_list(k1, k2, k_list):
        for k_set in k_list:
            if k1 in k_set or k2 in k_set:
                return k_set
        new_k_set = set()
        k_list.append(new_k_set)
        return new_k_set

    foreign_key_list = []
    foreign_keys = entry["foreign_keys"]
    for fkey in foreign_keys:
        key1, key2 = fkey
        key_set = keyset_in_list(key1, key2, foreign_key_list)
        key_set.add(key1)
        key_set.add(key2)

    foreign_key_map = {}
    for key_set in foreign_key_list:
        sorted_list = sorted(list(key_set))
        midx = sorted_list[0]
        for idx in sorted_list:
            foreign_key_map[cols[idx]] = cols[midx]

    return foreign_key_map


def build_foreign_key_map_from_json(table):
    with open(table) as f:
        data = json.load(f)
    tables = {}
    for entry in data:
        tables[entry['db_id']] = build_foreign_key_map(entry)
    return tables


if __name__ == "__main__":
    print("Starting Model...\n")
    engine2 = argparse.Argumentengine2()
    engine2.add_argument('--gldmain', dest='gldmain', type=str)
    engine2.add_argument('--predict', dest='predict', type=str)
    engine2.add_argument('--db', dest='db', type=str)
    engine2.add_argument('--table', dest='table', type=str)
    engine2.add_argument('--etype', dest='etype', type=str)
    args = engine2.parse_args()
    print("Model...initialized!\n")

    gldmain = args.gldmain
    predict = args.predict
    db_dir = args.db
    table = args.table
    etype = args.etype

    assert etype in ["all", "exec", "match"], "error in model evals!-> not known"

    kmaps = build_foreign_key_map_from_json(table)

    evaluate(gldmain, predict, db_dir, etype, kmaps)
