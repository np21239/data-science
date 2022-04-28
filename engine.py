import json
import sqlite3
from nltk import word_loadData
from sqlProperties import * 
class mapping:
    def __init__(self, mapping):
        self._mapping = mapping
        self._map_load = self._map(self._mapping)
    @property
    def mapping(self):
        return self._mapping
    @property
    def map_load(self):
        return self._map_load

    def _map(self, mapping):
        map_load = {'*': "__all__"}
        id = 1
        for key, valuess in mapping.items():
            for val in valuess:
                map_load[key.lower() + "." + val.lower()] = "__" + key.lower() + "." + val.lower() + "__"
                id += 1
        for key in mapping:
            map_load[key.lower()] = "__" + key.lower() + "__"
            id += 1
        return map_load


def get_mapping(db):
     mapping = {}
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [str(table[0].lower()) for table in cursor.fetchall()]
    for table in tables:
        cursor.execute("PRAGMA table_info({})".format(table))
        mapping[table] = [str(col[1].lower()) for col in cursor.fetchall()]
    return mapping


def get_mapping_from_json(fpath):
    with open(fpath) as f:
        data = json.load(f)
    mapping = {}
    for entry in data:
        table = str(entry['table'].lower())
        cols = [str(col['column_name'].lower()) for col in entry['col_data']]
        mapping[table] = cols

    return mapping

def loadAlsTBL(tokenized_words):
    as_ipxs = [ipx for ipx, tok in enumerate(tokenized_words) if tok == 'as']
    alias = {}
    for ipx in as_ipxs:
        alias[tokenized_words[ipx+1]] = tokenized_words[ipx-1]
    return alias
def loadData(string):
    string = str(string)
    string = string.replace("\'", "\"")  # ensures all string values wrapped by "" problem??
    quote_ipxs = [ipx for ipx, char in enumerate(string) if char == '"']
    assert len(quote_ipxs) % 2 == 0, "Unexpected quote"
    valuess = {}
    for i in range(len(quote_ipxs)-1, -1, -2):
        qipx1 = quote_ipxs[i-1]
        qipx2 = quote_ipxs[i]
        val = string[qipx1: qipx2+1]
        key = "__val_{}_{}__".format(qipx1, qipx2)
        string = string[:qipx1] + key + string[qipx2+1:]
        valuess[key] = val
    tokenized_words = [word.lower() for word in word_loadData(string)]
    for i in range(len(tokenized_words)):
        if tokenized_words[i] in valuess:
            tokenized_words[i] = valuess[tokenized_words[i]]
    eq_ipxs = [ipx for ipx, tok in enumerate(tokenized_words) if tok == "="]
    eq_ipxs.reverse()
    prefix = ('!', '>', '<')
    for eq_ipx in eq_ipxs:
        pre_tok = tokenized_words[eq_ipx-1]
        if pre_tok in prefix:
            tokenized_words = tokenized_words[:eq_ipx-1] + [pre_tok + "="] + tokenized_words[eq_ipx+1: ]

    return tokenized_words


def get_alias_words(mapping, tokenized_words):
    tables = loadAlsTBL(tokenized_words)
    for key in mapping:
        assert key not in tables, "Alias {} has the same name in table".format(key)
        tables[key] = key
    return tables


def loadd_col(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    tok = tokenized_words[start_ipx]
    if tok == "*":
        return start_ipx + 1, mapping.map_load[tok]

    if '.' in tok:  # if token is a composite
        alias, col = tok.split('.')
        key = alias_words[alias] + "." + col
        return start_ipx+1, mapping.map_load[key]

    assert dflt_tbl is not None and len(dflt_tbl) > 0, "Default tables should not be None or empty"

    for alias in dflt_tbl:
        table = alias_words[alias]
        if tok in mapping.mapping[table]:
            key = table + "." + tok
            return start_ipx+1, mapping.map_load[key]

    assert False, "Error col: {}".format(tok)


def loadd_col_unit(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    ipx = start_ipx
    len_ = len(tokenized_words)
    isBlock = False
    isDistinct = False
    if tokenized_words[ipx] == '(':
        isBlock = True
        ipx += 1

    if tokenized_words[ipx] in addtnl:
        agg_id = addtnl.index(tokenized_words[ipx])
        ipx += 1
        assert ipx < len_ and tokenized_words[ipx] == '('
        ipx += 1
        if tokenized_words[ipx] == "distinct":
            ipx += 1
            isDistinct = True
        ipx, col_id = loadd_col(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        assert ipx < len_ and tokenized_words[ipx] == ')'
        ipx += 1
        return ipx, (agg_id, col_id, isDistinct)

    if tokenized_words[ipx] == "distinct":
        ipx += 1
        isDistinct = True
    agg_id = addtnl.index("none")
    ipx, col_id = loadd_col(tokenized_words, ipx, alias_words, mapping, dflt_tbl)

    if isBlock:
        assert tokenized_words[ipx] == ')'
        ipx += 1  # skip ')'

    return ipx, (agg_id, col_id, isDistinct)


def loadd_main_unit(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    ipx = start_ipx
    len_ = len(tokenized_words)
    isBlock = False
    if tokenized_words[ipx] == '(':
        isBlock = True
        ipx += 1

    col_unit1 = None
    col_unit2 = None
    unit_op = ups.index('none')

    ipx, col_unit1 = loadd_col_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    if ipx < len_ and tokenized_words[ipx] in ups:
        unit_op = ups.index(tokenized_words[ipx])
        ipx += 1
        ipx, col_unit2 = loadd_col_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)

    if isBlock:
        assert tokenized_words[ipx] == ')'
        ipx += 1  # skip ')'

    return ipx, (unit_op, col_unit1, col_unit2)

def loadd_table_unit(tokenized_words, start_ipx, alias_words, mapping):
    ipx = start_ipx
    len_ = len(tokenized_words)
    key = alias_words[tokenized_words[ipx]]

    if ipx + 1 < len_ and tokenized_words[ipx+1] == "as":
        ipx += 3
    else:
        ipx += 1

    return ipx, mapping.map_load[key], key

def create_condition(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    ipx = start_ipx
    len_ = len(tokenized_words)
    conds = []

    while ipx < len_:
        ipx, main_unit = loadd_main_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        not_operator = False
        if tokenized_words[ipx] == 'not':
            not_operator = True
            ipx += 1

        assert ipx < len_ and tokenized_words[ipx] in sqlwhre, "Error condition: ipx: {}, tok: {}".format(ipx, tokenized_words[ipx])
        op_id = sqlwhre.index(tokenized_words[ipx])
        ipx += 1
        val1 = val2 = None
        if op_id == sqlwhre.index('between'):  # between..and... special case: dual values
            ipx, val1 = loadd_value(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
            assert tokenized_words[ipx] == 'and'
            ipx += 1
            ipx, val2 = loadd_value(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        else:  # normal case: single value
            ipx, val1 = loadd_value(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
            val2 = None

        conds.append((not_operator, op_id, main_unit, val1, val2))

        if ipx < len_ and (tokenized_words[ipx] in kwds or tokenized_words[ipx] in (")", ";") or tokenized_words[ipx] in sqljoin):
            break

        if ipx < len_ and tokenized_words[ipx] in conditnl:
            conds.append(tokenized_words[ipx])
            ipx += 1  # skip and/or

    return ipx, conds

def loadd_value(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    ipx = start_ipx
    len_ = len(tokenized_words)
    isBlock = False
    if tokenized_words[ipx] == '(':
        isBlock = True
        ipx += 1

    if tokenized_words[ipx] == 'select':
        ipx, val = loadd_sql(tokenized_words, ipx, alias_words, mapping)
    elif "\"" in tokenized_words[ipx]:  # token is a string value
        val = tokenized_words[ipx]
        ipx += 1
    else:
        try:
            val = float(tokenized_words[ipx])
            ipx += 1
        except:
            end_ipx = ipx
            while end_ipx < len_ and tokenized_words[end_ipx] != ',' and tokenized_words[end_ipx] != ')'\
                and tokenized_words[end_ipx] != 'and' and tokenized_words[end_ipx] not in kwds and tokenized_words[end_ipx] not in sqljoin:
                    end_ipx += 1

            ipx, val = loadd_col_unit(tokenized_words[start_ipx: end_ipx], 0, alias_words, mapping, dflt_tbl)
            ipx = end_ipx

    if isBlock:
        assert tokenized_words[ipx] == ')'
        ipx += 1
    return ipx, val
def loadSelect(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl=None):
    ipx = start_ipx
    len_ = len(tokenized_words)

    assert tokenized_words[ipx] == 'select', "'select' not found"
    ipx += 1
    isDistinct = False
    if ipx < len_ and tokenized_words[ipx] == 'distinct':
        ipx += 1
        isDistinct = True
    main_units = []

    while ipx < len_ and tokenized_words[ipx] not in kwds:
        agg_id = addtnl.index("none")
        if tokenized_words[ipx] in addtnl:
            agg_id = addtnl.index(tokenized_words[ipx])
            ipx += 1
        ipx, main_unit = loadd_main_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        main_units.append((agg_id, main_unit))
        if ipx < len_ and tokenized_words[ipx] == ',':
            ipx += 1  # skip ','
    return ipx, (isDistinct, main_units)

def loadWhere(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl):
    ipx = start_ipx
    len_ = len(tokenized_words)
    if ipx >= len_ or tokenized_words[ipx] != 'where':
        return ipx, []
    ipx += 1
    ipx, conds = create_condition(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    return ipx, conds
def loadd_from(tokenized_words, start_ipx, alias_words, mapping):
    assert 'from' in tokenized_words[start_ipx:], "'from' not found"
    len_ = len(tokenized_words)
    ipx = tokenized_words.index('from', start_ipx) + 1
    dflt_tbl = []
    table_units = []
    conds = []
    while ipx < len_:
        isBlock = False
        if tokenized_words[ipx] == '(':
            isBlock = True
            ipx += 1
        if tokenized_words[ipx] == 'select':
            ipx, sql = loadd_sql(tokenized_words, ipx, alias_words, mapping)
            table_units.append((tbltp['sql'], sql))
        else:
            if ipx < len_ and tokenized_words[ipx] == 'join':
                ipx += 1  # skip join
            ipx, table_unit, table_name = loadd_table_unit(tokenized_words, ipx, alias_words, mapping)
            table_units.append((tbltp['table_unit'],table_unit))
            dflt_tbl.append(table_name)
        if ipx < len_ and tokenized_words[ipx] == "on":
            ipx += 1  # skip on
            ipx, this_conds = create_condition(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
            if len(conds) > 0:
                conds.append('and')
            conds.extend(this_conds)
        if isBlock:
            assert tokenized_words[ipx] == ')'
            ipx += 1
        if ipx < len_ and (tokenized_words[ipx] in kwds or tokenized_words[ipx] in (")", ";")):
            break
    return ipx, table_units, conds, dflt_tbl


def create_grp_by(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl):
    ipx = start_ipx
    len_ = len(tokenized_words)
    col_units = []
    if ipx >= len_ or tokenized_words[ipx] != 'group':
        return ipx, col_units
    ipx += 1
    assert tokenized_words[ipx] == 'by'
    ipx += 1
    while ipx < len_ and not (tokenized_words[ipx] in kwds or tokenized_words[ipx] in (")", ";")):
        ipx, col_unit = loadd_col_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        col_units.append(col_unit)
        if ipx < len_ and tokenized_words[ipx] == ',':
            ipx += 1  # skip ','
        else:
            break
    return ipx, col_units

def loadd_having(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl):
    ipx = start_ipx
    len_ = len(tokenized_words)
    if ipx >= len_ or tokenized_words[ipx] != 'having':
        return ipx, []
    ipx += 1
    ipx, conds = create_condition(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    return ipx, conds


def loadd_order_by(tokenized_words, start_ipx, alias_words, mapping, dflt_tbl):
    ipx = start_ipx
    len_ = len(tokenized_words)
    main_units = []
    order_type = 'asc'
    if ipx >= len_ or tokenized_words[ipx] != 'order':
        return ipx, main_units
    ipx += 1
    assert tokenized_words[ipx] == 'by'
    ipx += 1
    while ipx < len_ and not (tokenized_words[ipx] in kwds or tokenized_words[ipx] in (")", ";")):
        ipx, main_unit = loadd_main_unit(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
        main_units.append(main_unit)
        if ipx < len_ and tokenized_words[ipx] in odps:
            order_type = tokenized_words[ipx]
            ipx += 1
        if ipx < len_ and tokenized_words[ipx] == ',':
            ipx += 1  # skip ','
        else:
            break

    return ipx, (order_type, main_units)


def loadSQL(mapping, query):
    tokenized_words = loadData(query)
    alias_words = get_alias_words(mapping.mapping, tokenized_words)
    _, sql = loadd_sql(tokenized_words, 0, alias_words, mapping)

    return sql


def skip_semicolon(tokenized_words, start_ipx):
    ipx = start_ipx
    while ipx < len(tokenized_words) and tokenized_words[ipx] == ";":
        ipx += 1
    return ipx

def loadd_limit(tokenized_words, start_ipx):
    ipx = start_ipx
    len_ = len(tokenized_words)

    if ipx < len_ and tokenized_words[ipx] == 'limit':
        ipx += 2
        return ipx, int(tokenized_words[ipx-1])

    return ipx, None

def loadd_sql(tokenized_words, start_ipx, alias_words, mapping):
    isBlock = False # indicate whether this is a block of sql/sub-sql
    len_ = len(tokenized_words)
    ipx = start_ipx

    sql = {}
    if tokenized_words[ipx] == '(':
        isBlock = True
        ipx += 1
    from_end_ipx, table_units, conds, dflt_tbl = loadd_from(tokenized_words, start_ipx, alias_words, mapping)
    sql['from'] = {'table_units': table_units, 'conds': conds}
    _, select_col_units = loadSelect(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    sql['select'] = select_col_units
    ipx, where_conds = loadWhere(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    sql['where'] = where_conds
    ipx, group_col_units = create_grp_by(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    sql['groupBy'] = group_col_units
    ipx, having_conds = loadd_having(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    sql['having'] = having_conds
    ipx, order_col_units = loadd_order_by(tokenized_words, ipx, alias_words, mapping, dflt_tbl)
    sql['orderBy'] = order_col_units
    ipx, limit_val = loadd_limit(tokenized_words, ipx)
    sql['limit'] = limit_val
    ipx = skip_semicolon(tokenized_words, ipx)
    if isBlock:
        assert tokenized_words[ipx] == ')'
        ipx += 1
    ipx = skip_semicolon(tokenized_words, ipx)
    for op in opertns:
        sql[op] = None
    if ipx < len_ and tokenized_words[ipx] in opertns:
        sql_op = tokenized_words[ipx]
        ipx += 1
        ipx, IUE_sql = loadd_sql(tokenized_words, ipx, alias_words, mapping)
        sql[sql_op] = IUE_sql
    return ipx, sql


def load_data(fpath):
    with open(fpath) as f:
        data = json.load(f)
    return data



