import sys
import os
from Bio import Phylo
from io import BytesIO     # for handling byte strings
from io import StringIO    # for handling unicode strings
from .Bandelt_Node import Bandelt_Node

def create_Bandelt_Tree(clade, parent_node, file_path, file_name, mapping_dic_dic):
    for idx in range(len(clade)):
        if idx == 0:
            if clade[idx].is_terminal():
                children_node = Bandelt_Node(int(mapping_dic_dic[os.path.join(file_path, file_name+'.nex')][clade[idx].name]))
#                 # Parent add 
                parent_node.left = children_node
                children_node.parent = parent_node
            else:
                children_node = Bandelt_Node(int(-100))
                parent_node.left = children_node
                children_node.parent = parent_node
        if idx == 1:  
            if clade[idx].is_terminal():
                children_node = Bandelt_Node(int(mapping_dic_dic[os.path.join(file_path, file_name+'.nex')][clade[idx].name]))
#                 # Parent add 
                parent_node.right = children_node
                children_node.parent = parent_node
            else:
                children_node = Bandelt_Node(int(-100))
                parent_node.right = children_node
                children_node.parent = parent_node
        create_Bandelt_Tree(clade[idx], children_node, file_path, file_name, mapping_dic_dic)
    
    
def find_Bandelt_encode(target_node, val):
    visited_nodes = []
    queue = []
    visited_nodes.append(target_node.data)
    queue.append(target_node)
    encode_num = None
    while queue:
        current_node = queue.pop(0)
        if current_node != None:
            if current_node.data != '*' and abs(int(current_node.data)) < abs(int(val)):
                encode_num = current_node.data
        if current_node.left != None:
            if current_node.left.data not in visited_nodes:
                visited_nodes.append(current_node.left.data)
                queue.append(current_node.left)
                
        if current_node.right != None:
            if current_node.right.data not in visited_nodes:
                visited_nodes.append(current_node.right.data)
                queue.append(current_node.right)   
        if encode_num != None:
            break
    return encode_num


def inner_node_indexation(current_node):
#     print(current_node.data)
    if current_node.left != None:
        left_data = inner_node_indexation(current_node.left)
    if current_node.right != None:
        right_data = inner_node_indexation(current_node.right)
    
    if current_node.left != None and current_node.right != None:
        # This is the inner node
        if len(left_data) == 0:
            current_node.data = int(-right_data[0])
            return []
        if len(right_data) == 0:
            current_node.data = int(-left_data[0])
            return []
        if len(left_data) != 0 and len(right_data) != 0:
            if abs(left_data[0]) > abs(right_data[0]):
                current_node.data = int(-left_data[0])
                return [right_data[0]]
            elif abs(left_data[0]) < abs(right_data[0]):
                current_node.data = int(-right_data[0])
                return [left_data[0]]
    elif current_node.left != None and current_node.right == None:
        pass
#         print("Finish!")
    else:
        # This is the terminal node
        return [current_node.data]


def Bandelt_Encode(file_path, file_name, mapping_dic_dic):
    tree = Phylo.read(os.path.join(file_path, file_name+'.nex.treefile'), 'newick')
#     Phylo.draw(tree)
    
    # This is the root
    root_node = Bandelt_Node(sys.maxsize)
    inner_root = Bandelt_Node(-100)
    
    if tree.root[1].is_terminal():
        left_node = Bandelt_Node(int(mapping_dic_dic[os.path.join(file_path, file_name+'.nex')][tree.root[1].name]))
        inner_root.left = left_node
        left_node.parent = inner_root
    else:
        left_node = Bandelt_Node(int(-100))
        inner_root.left = left_node
        left_node.parent = inner_root

    if tree.root[2].is_terminal():
        right_node = Bandelt_Node(int(mapping_dic_dic[os.path.join(file_path, file_name+'.nex')][tree.root[2].name]))
        inner_root.right = right_node
        right_node.parent = inner_root
    else:
        right_node = Bandelt_Node(int(-100))
        inner_root.right = right_node
        right_node.parent = inner_root
        
    # Add linkage for the first three node
    root_node.left = inner_root
    inner_root.parent = root_node
    
    create_Bandelt_Tree(tree.root[1], left_node, file_path, file_name, mapping_dic_dic)
    create_Bandelt_Tree(tree.root[2], right_node, file_path, file_name, mapping_dic_dic)
    
    inner_node_indexation(root_node)
    
    encode_num_list = []
    for i in range(0, tree.count_terminals()-2):
        target_node_val = -(i+1)
        found_node = root_node.find_node(target_node_val)
        encode_num = find_Bandelt_encode(found_node, target_node_val)
        encode_num_list.append(encode_num)
    return (root_node, encode_num_list)