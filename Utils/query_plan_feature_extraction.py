import copy
import hashlib
import os
import random
import re
import string
from datetime import datetime
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from functools import lru_cache
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

def load_database_info(dbname):
    file_path = f'../Data/{dbname}/database_statistics/'
    tables_index = np.load(file_path + "tables_index.npy", allow_pickle=True).item()
    tables_index_all = np.load(file_path + "tables_index_all.npy", allow_pickle=True).item()
    columns_index = np.load(file_path + "columns_index.npy", allow_pickle=True).item()
    columns_list = np.load(file_path + "columns_list.npy", allow_pickle=True)
    attribute_range = np.load(file_path + "attribute_range.npy", allow_pickle=True).item()
    nodes = np.load(file_path + "postgresql_nodestypes_all.npy", allow_pickle=True).item()
    return tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes

def replace_aliases_and_columns(original_plan, columns_list):
    alias_map = {}
    def collect_aliases(node):
        if "Relation Name" in node and "Alias" in node:
            alias_map[node["Alias"]] = node["Relation Name"]
        if "Plans" in node:
            for subplan in node["Plans"]:
                collect_aliases(subplan)

    def apply_aliases(node, table_name=None):
        new_node = copy.deepcopy(node)
        current_table = new_node.get("Relation Name", table_name)

        alias_patterns = {alias: re.compile(r'\\b' + re.escape(alias) + r'\\.') for alias in alias_map}

        for key, value in new_node.items():
            if isinstance(value, str):
                for alias, full_name in alias_map.items():
                    pat = alias_patterns[alias]
                    if pat.search(value):
                        value = pat.sub(full_name + ".", value)
                        new_node[key] = value

        if "Plans" in new_node:
            new_node["Plans"] = [apply_aliases(subplan, current_table) for subplan in new_node["Plans"]]

        format_column_names(new_node, current_table)
        return new_node

    def format_column_names(node, table_name):
        if table_name:
            columns_pattern = re.compile(r'\\b(' + '|'.join(re.escape(column) for column in columns_list) + r')\\b')
            for key, value in node.items():
                if isinstance(value, str):
                    def replace_columns(match):
                        column_name = match.group(0)
                        full_pattern = re.compile(r'(?<![\\w.])' + re.escape(column_name) + r'(?![\\w.])')
                        if full_pattern.search(value):
                            before = value[:match.start()]
                            after = value[match.end():]
                            if not (before.endswith('.') or re.match(r'\\.\\s*\\w+', after)):
                                return f"{table_name}.{column_name}"
                        return column_name
                    new_value = columns_pattern.sub(replace_columns, value)
                    if new_value != value:
                        node[key] = new_value

    collect_aliases(original_plan)
    new_plan = apply_aliases(original_plan)
    return new_plan

def replace_aliases_and_columns_in_query_paln(data, columns_list):
    data_replaced = []
    for query_plan in data:
        data_replaced.append(replace_aliases_and_columns(query_plan, columns_list))
    return replace_aliases_and_columns(data, columns_list)

def extract_predicates(text):
    predicate_patterns = [
        r'(\w+\.\w+)\s*([=<>]{1,2}|<>|~~|!~~|in|like|not like)\s*(\{.*?\}|\[.*?\]|".*?"|\'[^\']*?\'|\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?|\S+)'
    ]
    predicates = []
    segments = text.split('|')
    for segment in segments:
        for pattern in predicate_patterns:
            matches = re.findall(pattern, segment)
            for match in matches:
                if len(match) == 3:
                    table_column, operator, raw_value = match
                    value = clean_value(raw_value)
                    predicates.append([table_column, operator, value])
                else:
                    print("Unexpected match format:", match)
    return predicates

def clean_value(value):
    value = value.strip("',\")")
    if value.startswith('{') and value.endswith('}'):
        return '{' + re.sub(r"['\"]", "", value[1:-1]) + '}'
    value = re.sub(r'::.*', '', value)
    return value

def is_float_num(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

@lru_cache(maxsize=32)
def _load_w2v_cached(model_path):
    return Word2Vec.load(model_path)

def str_value_encoding(values, model_path):
    model = _load_w2v_cached(model_path)
    translation_table = str.maketrans('', '', string.punctuation)
    processed_values = [value.lower().translate(translation_table).split() for value in values]
    words = [word for sublist in processed_values for word in sublist]
    try:
        word_vectors = [model.wv[word] for word in words if word in model.wv]
        if word_vectors:
            encoded_value = np.mean(word_vectors, axis=0)[0]
        else:
            encoded_value = -2
    except Exception as e:
        print(f'Word2vec model {model_path} error or empty input!', e)
        encoded_value = -2
    return encoded_value

def str_to_unit_float(s: str) -> float:
    digest = hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest()
    n = int.from_bytes(digest, 'big')
    return n / (1 << 64)

def Text_extraction(text, tables_index, tables_index_all, columns_index, attribute_range):
    enc_column = [0]*(int(len(columns_index)*8))
    enc_table = [0] * len(tables_index)
    predicates = extract_predicates(text)

    for p in predicates:
        if p[0] in columns_index:
            if p[2] in columns_index:
                enc_column[columns_index[p[0]]*8] = 1
                enc_column[columns_index[p[2]]*8] = 1
                operators = ['join', '=', '<', '<=', '>', '>=', '<>', 'in']
                if p[1] != '=':
                    op_i = operators.index(p[1])
                    enc_column[columns_index[p[0]]*8 + op_i] = 1
                    enc_column[columns_index[p[2]]*8 + op_i] = 1
            else:
                v = attribute_range[p[0]]
                if len(v) == 3:
                    operators = ['join', '=', '<', '<=', '>', '>=', '<>', 'in']
                    if p[2] == 'ANY':
                        enc_column[columns_index[p[0]] * 8 + operators.index(p[1])] = 2
                        continue
                    if p[2] == '(SubPlan':
                        enc_column[columns_index[p[0]] * 8 + operators.index(p[1])] = 0.5
                        continue

                    if type(v[0]) != datetime:
                        r = v[1] - v[0]
                        if r == 0:
                            r = 1
                        p_v = float(re.findall(r'-?\d+\.?\d*e?-?\d*?', p[2])[0])
                    else:
                        r = v[1] - v[0]
                        p_v = ' '.join(p[2:]).split('::')[0].replace('\'', '')
                        try:
                            p_v = datetime.strptime(p_v, '%Y-%m-%d')
                        except:
                            p_v = datetime.strptime(p_v, '%Y-%m-%d %H:%M:%S')
                    if '{' in p[2] and p[1] == '=':
                        in_num = 0
                        num = 0
                        L_p_v = p[2].split('}')[0][2:].split(',')
                        for p_v in L_p_v:
                            p_v = float(p_v)
                            if p_v < v[0] or p_v > v[1]:
                                continue
                            else:
                                num += 1 + (p_v - v[0]) / r
                                in_num += 1
                        enc_column[columns_index[p[0]]*8 + 1] = num/in_num
                        enc_column[columns_index[p[0]]*8 + 7] = 1 + in_num * v[2]
                        continue

                    if p[1] == '<':
                        if p_v <= v[0]:
                            num = -1
                        elif p_v > v[1]:
                            num = 2
                        else:
                            num = 1 + (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 2] = num
                    elif p[1] == '<=':
                        if p_v < v[0]:
                            num = -1
                        elif p_v >= v[1]:
                            num = 2
                        else:
                            num = 1 + (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 3] = num
                    elif p[1] == '>':
                        if p_v >= v[1]:
                            num = -1
                        elif p_v < v[0]:
                            num = 2
                        else:
                            num = 2 - (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 4] = num
                    elif p[1] == '>=':
                        if p_v > v[1]:
                            num = -1
                        elif p_v <= v[0]:
                            num = 2
                        else:
                            num = 2 - (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 5] = num
                    elif p[1] == '=':
                        if p_v < v[0] or p_v > v[1]:
                            num = -1
                        else:
                            num = 1 + (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 1] = num
                    elif p[1] == '<>':
                        if p_v < v[0] or p_v > v[1]:
                            num = 2
                        else:
                            num = 1 + (p_v - v[0]) / r
                        enc_column[columns_index[p[0]]*8 + 6] = num
                else:
                    operators = ['join', '=', '>', '<', '~~', '!~~', '<>', 'in']
                    if p[2] == 'ANY':
                        enc_column[columns_index[p[0]] * 8 + operators.index(p[1])] = 2
                        continue

                    if '{' in p[2] and p[1] == '=':
                        in_num = 0
                        num = 0
                        L_p_v = ' '.join(p[2:]).split('}')[0][2:].split(',')
                        for p_v in L_p_v:
                            p_v = p_v.split(' ')
                            while '' in p_v:
                                p_v.remove('')
                            str_vec = str_value_encoding(p_v, v[0])
                            if str_vec == -2:
                                continue
                            else:
                                num += str_vec
                                in_num += 1
                        enc_column[columns_index[p[0]]*8 + 1] = num/in_num
                        enc_column[columns_index[p[0]]*8 + 7] = 1 + in_num * v[1]
                    else:
                        p_v = ' '.join(p[2:]).split('::')[0][1:-1].split('%')
                        while '' in p_v:
                            p_v.remove('')
                        if v[0] != 'None':
                            str_vec = str_value_encoding(p_v, v[0])
                        else:
                            str_vec = str_to_unit_float(' '.join(p_v))
                        op_i = operators.index(p[1])
                        enc_column[columns_index[p[0]]*8 + op_i] = str_vec
        else:
            continue

    words = re.findall(r'\\b\\w+\\b', text)
    for item in words:
        if item in tables_index_all:
            enc_table[tables_index_all[item]] = 1

    return enc_table + enc_column

def Node_info_extract(node, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats):
    nodetype_enc = [0] * len(nodes)
    if node['Node Type'] in nodes:
        nodetype_enc[nodes[node['Node Type']]] = 1

    info = ""
    if node['Node Type'] == "Seq Scan":
        if "Filter" in node:
            info += node["Filter"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]

    elif node['Node Type'] == "Hash":
        pass

    elif node['Node Type'] == "Hash Join":
        if "Hash Cond" in node:
            info += str(node["Hash Cond"])
        if "Join Filter" in node:
            info += "|" + str(node["Join Filter"])

    elif node['Node Type'] == "Sort":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        else:
            pass

    elif node['Node Type'] == "Aggregate":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "Gather Merge":
        pass

    elif node['Node Type'] == "CTE Scan":
        if "CTE Name" in node:
            info += node["CTE Name"]
        if "Filter" in node:
            info += "|" + node["Filter"]

    elif node['Node Type'] == "Nested Loop":
        if "Join Filter" in node:
            info += str(node["Join Filter"])
        else:
            pass

    elif node['Node Type'] == "Index Scan":
        if "Filter" in node:
            info += node["Filter"]
        if "Index Name" in node:
            info += "|" + node["Index Name"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]
        if "Index Cond" in node:
            info += "|" + node["Index Cond"]

    elif node['Node Type'] == "Limit":
        pass

    elif node['Node Type'] == "Append":
        pass

    elif node['Node Type'] == "Merge Join":
        if "Merge Cond" in node:
            info += node["Merge Cond"]
        if "Join Filter" in node:
            info += "|" + node["Join Filter"]

    elif node['Node Type'] == "Bitmap Index Scan":
        if "Index Name" in node:
            info += "|" + node["Index Name"]
        if "Index Cond" in node:
            info += "|" + node["Index Cond"]

    elif node['Node Type'] == "Bitmap Heap Scan":
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]
        if "Alias" in node:
            info += "|" + node["Alias"]
        if "Recheck Cond" in node:
            info += "|" + node["Recheck Cond"]

    elif node['Node Type'] == "Unique":
        pass

    elif node['Node Type'] == "Gather":
        pass

    elif node['Node Type'] == "Materialize":
        pass

    elif node['Node Type'] == "Subquery Scan":
        pass

    elif node['Node Type'] == "SetOp":
        pass

    elif node['Node Type'] == "WindowAgg":
        pass

    elif node['Node Type'] == "Memoize":
        if "Cache Key" in node:
            info += str(node["Cache Key"])
        else:
            pass

    elif node['Node Type'] == "Index Only Scan":
        if "Index Name" in node:
            info += node["Index Name"]
        if "Relation Name" in node:
            info += "|" + node["Relation Name"]
        if "Index Cond" in node:
            info += "|" + node["Index Cond"]

    elif node['Node Type'] == "Incremental Sort":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        if "Presorted Key" in node:
            info += "|" + str(node["Presorted Key"])

    elif node['Node Type'] == "Group":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "Result":
        pass

    elif node['Node Type'] == "Merge Append":
        if "Sort Key" in node:
            info += str(node["Sort Key"])
        else:
            pass

    elif node['Node Type'] == "HashAggregate":
        if "Group Key" in node:
            info += str(node["Group Key"])
        else:
            pass

    elif node['Node Type'] == "BitmapAnd":
        pass

    else:
        print('Unknown Node Type:', node['Node Type'])

    stats_enc = [(np.log(node["Total Cost"] + 1) - query_plans_stats[1][0])/(query_plans_stats[2][0] - query_plans_stats[1][0]), (np.log(node["Plan Rows"] + 1) - query_plans_stats[1][1])/(query_plans_stats[2][1] - query_plans_stats[1][1])]

    filter_enc = Text_extraction(info, tables_index, tables_index_all, columns_index, attribute_range)

    return nodetype_enc + stats_enc + filter_enc

def Subtree_traversal(tree, L, index):
    node_index = index
    if 'Plans' in tree:
        for i in range(len(tree['Plans'])):
            L, index = Subtree_traversal(tree['Plans'][i], L, index+1)
    if 'Plans' in tree and len(tree['Plans']) >= 2:
        L.append(tree)
        if node_index == 0:
            return L
    else:
        if node_index == 0:
            if L != []:
                L[-1] = tree
                return L
            else:
                return [tree]
    return L, index

def Data_augmentation(data):
    data_plus = []
    for tree in data:
        L = Subtree_traversal(tree, [], 0)
        if L != []:
            data_plus = data_plus + L
    return data_plus

def get_plan_stats(data):
    costs = []
    rows = []

    def recurse(n):
        costs.append(n["Total Cost"])
        rows.append(n["Plan Rows"])
        if "Plans" in n:
            for child in n["Plans"]:
                recurse(child)

    for plan in data:
        recurse(plan)

    costs = np.array(costs)
    rows = np.array(rows)

    costs = np.log(costs + 1)
    rows = np.log(rows + 1)

    costs_min = np.min(costs)
    costs_max = np.max(costs)
    rows_min = np.min(rows)
    rows_max = np.max(rows)

    return [["Total Cost", "Plan Rows"], [costs_min, rows_min], [costs_max, rows_max]]

def Join_only_tree(tree, join_tree):
    if 'Plans' in tree:
        child_num = len(tree['Plans'])
        if child_num == 1:
            if tree['Node Type'] in ["Hash Join", "Nested Loop", "Merge Join"]:
                join_tree = Join_only_tree(tree['Plans'][0], join_tree)
                if join_tree != {}:
                    join_tree = {'joins':[join_tree], 'label':tree["Actual Total Time"] * 1}
                else:
                    join_tree = {'label':tree["Actual Total Time"] * 1}
            else:
                join_tree = Join_only_tree(tree['Plans'][0], join_tree)
        else:
            join_tree_child = []
            for i in range(child_num):
                join_tree_child_i = Join_only_tree(tree['Plans'][i], join_tree)
                if join_tree_child_i != {}:
                    join_tree_child.append(join_tree_child_i)
            if join_tree_child != []:
                join_tree = {'joins': join_tree_child, 'label': tree["Actual Total Time"] * 1}
            else:
                join_tree = {'label': tree["Actual Total Time"] * 1}
    return join_tree

def Join_p_c_pair(join_tree, index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label):
    node_index = index
    if 'joins' in join_tree:
        if len(join_tree['joins']) == 1:
            L_pair_index.append([node_index, index+1])
            if (join_tree['label']-join_tree['joins'][0]['label']) < 0:
                L_pair_label.append(abs(join_tree['label']-join_tree['joins'][0]['label']))
                # print('error! join_tree label is smaller than its child!')
                # print(str(node_index) + ' ' + str(join_tree))
            else:
                L_pair_label.append(join_tree['label']-join_tree['joins'][0]['label'])
            index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label = Join_p_c_pair(join_tree['joins'][0], index+1, post_t_index, post_t_index_dic, L_pair_index, L_pair_label)
        else:
            multi_child_node_index = []
            for i in range(len(join_tree['joins'])):
                multi_child_node_index.append(index+1)
                index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label = Join_p_c_pair(join_tree['joins'][i], index+1, post_t_index, post_t_index_dic, L_pair_index, L_pair_label)
            L_pair_index.append([node_index, multi_child_node_index])
            if (join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))])) < 0:
                L_pair_label.append(abs(join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))])))
                # print('error! join_tree label is smaller than its multi child!')
                # print(str(node_index) + ' ' + str(join_tree))
            else:
                L_pair_label.append(join_tree['label']-sum([join_tree['joins'][i]['label'] for i in range(len(join_tree['joins']))]))
    else:
        L_pair_index.append([node_index, node_index])
        L_pair_label.append(join_tree['label'])
    post_t_index_dic[node_index] = post_t_index
    post_t_index += 1

    if node_index == 0:
        L_pair_index_post = []
        for i in L_pair_index:
            if type(i[1]) == int:
                L_pair_index_post.append([post_t_index_dic[i[0]], post_t_index_dic[i[1]]])
            else:
                L_pair_index_post.append([post_t_index_dic[i[0]], [post_t_index_dic[j] for j in i[1]]])
        return L_pair_index_post, L_pair_label
    return index, post_t_index, post_t_index_dic, L_pair_index, L_pair_label

def join_tree_correct(tree, L_label):
    if 'joins' in tree:
        if len(tree['joins']) == 1:
            subtree, L_label = join_tree_correct(tree['joins'][0], L_label)
            if tree['label'] < subtree['label']:
                if tree['label'] < tree['joins'][0]['label']:
                    tree['label'] = subtree['label'] + tree['label']
                else:
                    tree['label'] = subtree['label'] + tree['label'] - tree['joins'][0]['label']
            tree['joins'][0] = subtree
        else:
            subtree = []
            for i in range(len(tree['joins'])):
                subtree_i, L_label = join_tree_correct(tree['joins'][i], L_label)
                subtree.append(subtree_i)
            if tree['label'] < sum([subtree[i]['label'] for i in range(len(subtree))]):
                if tree['label'] < sum([tree['joins'][i]['label'] for i in range(len(tree['joins']))]):
                    tree['label'] = sum([subtree[i]['label'] for i in range(len(subtree))]) + tree['label']
                else:
                    tree['label'] = sum([subtree[i]['label'] for i in range(len(subtree))]) + tree['label'] - sum([tree['joins'][i]['label'] for i in range(len(tree['joins']))])
            for i in range(len(tree['joins'])):
                tree['joins'][i] = subtree[i]
    L_label.append(tree['label'])
    return tree, L_label

def get_join_tree_label(tree, L_label):
    if 'joins' in tree:
        if len(tree['joins']) == 1:
            L_label = join_tree_correct(tree['joins'][0], L_label)
        else:
            for i in range(len(tree['joins'])):
                L_label = join_tree_correct(tree['joins'][i], L_label)
    L_label.append(tree['label'])
    return L_label

def join_explanation_generate(tree):
    join_tree = Join_only_tree(tree, {})
    if join_tree == {}:
        join_tree = {'label': tree["Actual Total Time"] * 1}
    join_tree['label'] = tree["Actual Total Time"] * 1
    join_tree, L_label = join_tree_correct(join_tree, [])
    L_pair_index, L_pair_label = Join_p_c_pair(join_tree, 0, 0, {}, [], [])
    return L_pair_index, L_pair_label, L_label

def encoding_generate(tree, index, post_t_index, post_t_index_dic, L_n, L_e, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats):
    node_index = index
    if 'Plans' in tree:
        for i in range(len(tree['Plans'])):
            # Add edges in directed graph format; use current_index for clearer reference
            L_e.append([index + 1, node_index])
            index, L_n, L_e, post_t_index, post_t_index_dic = encoding_generate(tree['Plans'][i], index + 1, post_t_index, post_t_index_dic, L_n, L_e, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats)
    # After all children are processed, record the post-traversal index of this node
    post_t_index_dic[node_index] = post_t_index
    post_t_index += 1
    # Extract and append node features to L_n
    node_feature = Node_info_extract(tree, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats)
    L_n.append(node_feature)
    return index, L_n, L_e, post_t_index, post_t_index_dic

# ---- workers (top-level for pickling) ----
def _worker_generate_dataset(args):
    plans_i, plans_index_i, q_index_i, columns_list, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats = args
    dataset_i = []
    query_plans_index_new_i = []
    query_plans_postgres_cost_new_i = []
    for j in range(len(plans_i)):
        tree = replace_aliases_and_columns(plans_i[j], columns_list)
        __, node_feature, edge_index, __, post_t_index_dic = encoding_generate(tree, 0, 0, {}, [], [],
                                                                               tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats)
        edge_index = [[post_t_index_dic[e[0]], post_t_index_dic[e[1]]] for e in edge_index]
        if len(node_feature) < 2 or len(edge_index) < 1:
            print(f"Error in query plan {j}: too few nodes or edges")
            continue
        dataset_i.append([node_feature, edge_index, tree["Actual Total Time"]])
        query_plans_index_new_i.append(plans_index_i[j])
        query_plans_postgres_cost_new_i.append(tree["Total Cost"])
    keep = len(query_plans_index_new_i) >= 2
    return keep, q_index_i, dataset_i, query_plans_index_new_i, query_plans_postgres_cost_new_i

def _worker_generate_dataset_with_explanation(args):
    plans_i, plans_index_i, q_index_i, columns_list, tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats = args
    dataset_i = []
    query_plans_index_new_i = []
    query_plans_postgres_cost_new_i = []
    query_plans_subtree_postgres_cost_new_i = []
    dataset_subtree_labels_i = []
    dataset_labels_i = []
    dataset_sample_index_i = []
    dataset_join_pair_index_for_explain_i = []
    dataset_join_pair_label_for_explain_i = []
    sample_index_i = 0  # relative; will be offset in main

    for j in range(len(plans_i)):
        tree = replace_aliases_and_columns(plans_i[j], columns_list)
        subtrees = Data_augmentation([tree])
        subtrees_num = len(subtrees)
        if subtrees_num == 0:
            print(f"Error in query plan {j}: no subtrees generated")
            continue
        processed_subtrees = []
        for subtree in subtrees:
            __, node_feature, edge_index, __, post_t_index_dic = encoding_generate(subtree, 0, 0, {}, [], [],
                                                                                   tables_index, tables_index_all, columns_index, attribute_range, nodes, query_plans_stats)
            edge_index = [[post_t_index_dic[e[0]], post_t_index_dic[e[1]]] for e in edge_index]
            if len(node_feature) < 2 or len(edge_index) < 1:
                print(f"Error in query plan {j}: too few nodes or edges")
                continue
            processed_subtrees.append(Data(x=torch.FloatTensor(node_feature), edge_index=torch.LongTensor(edge_index).t()))
            query_plans_subtree_postgres_cost_new_i.append(subtree["Total Cost"])

        subtreeset_loader = DataLoader(dataset=processed_subtrees, batch_size=subtrees_num, shuffle=False)
        for _, batch in enumerate(subtreeset_loader):
            dataset_i.append([batch.x.float().tolist(), batch.edge_index.tolist(), sample_index_i])
            dataset_sample_index_i.append(batch.batch.tolist())
            sample_index_i += 1

        L_pair_index, L_pair_label, L_label = join_explanation_generate(tree)
        dataset_subtree_labels_i.append(L_label)
        dataset_labels_i += L_label
        dataset_join_pair_index_for_explain_i.append(L_pair_index)
        dataset_join_pair_label_for_explain_i.append(L_pair_label)
        query_plans_index_new_i.append(plans_index_i[j])
        query_plans_postgres_cost_new_i.append(tree["Total Cost"])

    keep = len(query_plans_index_new_i) >= 2
    return (keep, q_index_i, dataset_i, dataset_sample_index_i, query_plans_index_new_i,
            query_plans_postgres_cost_new_i, query_plans_subtree_postgres_cost_new_i,
            dataset_labels_i, dataset_subtree_labels_i, dataset_join_pair_index_for_explain_i,
            dataset_join_pair_label_for_explain_i, sample_index_i)

def generate_dataset(dbname):
    # Load database statistics
    tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes = load_database_info(dbname)
    # Load query plans
    workloads_path_base = f'../Data/{dbname}/workloads/postgresql_{dbname}_executed_query'
    query_plans = np.load(f'{workloads_path_base}_plans.npy', allow_pickle=True)
    query_plans_index = np.load(f'{workloads_path_base}_plans_index.npy', allow_pickle=True)
    query_index = np.load(f'{workloads_path_base}_index.npy', allow_pickle=True)

    # shuffle the workloads wit index in the same order
    c = list(zip(query_plans, query_plans_index, query_index))
    random.seed(0)
    random.shuffle(c)
    query_plans, query_plans_index, query_index = zip(*c)
    query_plans = list(query_plans)
    query_plans_index = list(query_plans_index)
    query_index = list(query_index)

    query_plans_flat = [item for sublist in query_plans for item in sublist]
    query_plans_stats = get_plan_stats(query_plans_flat)

    dataset = []
    query_index_new = []
    query_plans_index_new = []
    query_plans_postgres_cost_new = []

    # parallel per-query
    args_iter = zip(query_plans, query_plans_index, query_index,
                    repeat(columns_list), repeat(tables_index), repeat(tables_index_all),
                    repeat(columns_index), repeat(attribute_range), repeat(nodes), repeat(query_plans_stats))
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        for keep, q_idx, dataset_i, qp_idx_i, qp_cost_i in tqdm(ex.map(_worker_generate_dataset, args_iter, chunksize=4), total=len(query_plans)):
            if keep:
                dataset.extend(dataset_i)
                query_index_new.append(q_idx)
                query_plans_index_new.append(qp_idx_i)
                query_plans_postgres_cost_new.append(qp_cost_i)

    query_plans_index_num_new = [len(s) for s in query_plans_index_new]

    dataset_path_base = f'../Data/{dbname}/datasets/'
    os.makedirs(dataset_path_base, exist_ok=True)
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_dataset.npy', np.array(dataset, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_index.npy', np.array(query_index_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_index.npy', np.array(query_plans_index_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_index_num.npy', np.array(query_plans_index_num_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_postgres_cost.npy', np.array(query_plans_postgres_cost_new, dtype=object))

def generate_dataset_with_explanation(dbname):
    # Load database statistics
    tables_index, tables_index_all, columns_index, columns_list, attribute_range, nodes = load_database_info(dbname)
    # Load query plans
    workloads_path_base = f'../Data/{dbname}/workloads/postgresql_{dbname}_executed_query'
    query_plans = np.load(f'{workloads_path_base}_plans.npy', allow_pickle=True)
    query_plans_index = np.load(f'{workloads_path_base}_plans_index.npy', allow_pickle=True)
    query_index = np.load(f'{workloads_path_base}_index.npy', allow_pickle=True)

    # shuffle the workloads wit index in the same order
    c = list(zip(query_plans, query_plans_index, query_index))
    random.seed(0)
    random.shuffle(c)
    query_plans, query_plans_index, query_index = zip(*c)
    query_plans = list(query_plans)
    query_plans_index = list(query_plans_index)
    query_index = list(query_index)

    query_plans_flat = [item for sublist in query_plans for item in sublist]
    query_plans_stats = get_plan_stats(query_plans_flat)

    sample_index = 0
    dataset = []
    query_index_new = []
    query_plans_index_new = []
    query_plans_postgres_cost_new = []
    query_plans_subtree_postgres_cost_new = []

    dataset_labels = []
    dataset_sample_index = []
    dataset_subtree_labels = []
    dataset_join_pair_index_for_explain = []
    dataset_join_pair_label_for_explain = []

    # parallel per-query
    args_iter = zip(query_plans, query_plans_index, query_index,
                    repeat(columns_list), repeat(tables_index), repeat(tables_index_all),
                    repeat(columns_index), repeat(attribute_range), repeat(nodes), repeat(query_plans_stats))
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as ex:
        results = list(tqdm(ex.map(_worker_generate_dataset_with_explanation, args_iter, chunksize=4), total=len(query_plans)))

    for (keep, q_idx_i, dataset_i, dataset_sample_index_i, qp_idx_i,
         qp_cost_i, qp_subtree_cost_i, labels_i, subtree_labels_i,
         pair_idx_i, pair_label_i, rel_count) in results:
        if not keep:
            continue
        # fix relative sample_index to global
        for entry in dataset_i:
            entry[2] += sample_index
        dataset += dataset_i
        dataset_sample_index += dataset_sample_index_i
        sample_index += rel_count

        query_index_new.append(q_idx_i)
        query_plans_index_new.append(qp_idx_i)
        query_plans_postgres_cost_new.append(qp_cost_i)
        query_plans_subtree_postgres_cost_new += qp_subtree_cost_i
        dataset_labels += labels_i
        dataset_subtree_labels += subtree_labels_i
        dataset_join_pair_index_for_explain += pair_idx_i
        dataset_join_pair_label_for_explain += pair_label_i

    query_plans_index_num_new = [len(s) for s in query_plans_index_new]
    query_plans_subtrees_num = [len(s) for s in dataset_subtree_labels]

    dataset_path_base = f'../Data/{dbname}/datasets/'
    os.makedirs(dataset_path_base, exist_ok=True)
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_dataset_with_explanation.npy', np.array(dataset, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_index_with_explanation.npy', np.array(query_index_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_index_with_explanation.npy', np.array(query_plans_index_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_postgres_cost_with_explanation.npy', np.array(query_plans_postgres_cost_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_subtree_postgres_cost_with_explanation.npy', np.array(query_plans_subtree_postgres_cost_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_labels_with_explanation.npy', np.array(dataset_labels, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_sample_index_with_explanation.npy', np.array(dataset_sample_index, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_subtree_labels_with_explanation.npy', np.array(dataset_subtree_labels, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_join_pair_index_for_explain_with_explanation.npy', np.array(dataset_join_pair_index_for_explain, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_join_pair_label_for_explain_with_explanation.npy', np.array(dataset_join_pair_label_for_explain, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_index_num_with_explanation.npy', np.array(query_plans_index_num_new, dtype=object))
    np.save(f'{dataset_path_base}postgresql_{dbname}_executed_query_plans_subtrees_num_with_explanation.npy', np.array(query_plans_subtrees_num, dtype=object))

if __name__ == '__main__':
    dbname = 'stats'
    generate_dataset(dbname)
    # generate_dataset_with_explanation(dbname)



