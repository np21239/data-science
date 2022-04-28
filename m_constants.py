REM_VAL = True
REM_DISTN = True


CLS_WDS = ('select', 'from', 'where', 'group', 'order', 'limit', 'intersect', 'union', 'except')
EMBEDDING_KEYWDS = ('join', 'on', 'as')

OPERATIONAL_WHERE = ('not', 'between', '=', '>', '<', '>=', '<=', '!=', 'in', 'like', 'is', 'exists')
SIOPS = ('none', '-', '+', "*", '/')
ADDOPS = ('none', 'max', 'min', 'count', 'sum', 'avg')
TBTYP = {
    'sql': "sql",
    'table_unit': "table_unit",
}

CDPS = ('and', 'or')
SQLS = ('intersect', 'union', 'except')
ODRPS = ('desc', 'asc')


HDS = {
    "component1": ('where', 'group', 'order', 'limit', 'join', 'or', 'like'),
    "component2": ('except', 'union', 'intersect')
}
