import sys
import os
import copy
from .Bandelt_Node import Bandelt_Node

def post_order_traversal_num_2_name(current_node, file_path, file_base_name, mapping_dic_dic_decode):
    if current_node.left != None:
        left_newick_string = post_order_traversal_num_2_name(current_node.left, file_path, file_base_name, mapping_dic_dic_decode)
    if current_node.right != None:
        right_newick_string = post_order_traversal_num_2_name(current_node.right, file_path, file_base_name, mapping_dic_dic_decode)
    if current_node.data < 0:
        current_node.data = ''
#         pass
    elif current_node.data >= 0:
        current_node.data = mapping_dic_dic_decode[os.path.join(file_path, file_base_name+'.nex')][int(current_node.data)]
        

def post_order_traversal_newick_string(current_node):
#     print(current_node.data)
    if current_node.left != None:
        left_newick_string = post_order_traversal_newick_string(current_node.left)
    if current_node.right != None:
        right_newick_string = post_order_traversal_newick_string(current_node.right)

    if current_node.left == None and current_node.right == None:     
#         # This node is the terminal vertex
        return str(current_node.data)

    if current_node.left != None and current_node.right != None:
        newick_string = '(' + left_newick_string + ',' + right_newick_string + ')' + str(current_node.data)
        
    if current_node.left != None and current_node.right == None:
        newick_string = '(' + str(current_node.data) + ',' + left_newick_string + ')'

    return newick_string


def Bandelt_Decode(Bandelt_Encode_list, file_path, file_base_name, mapping_dic_dic_decode):
    BANDELT_NUM = len(Bandelt_Encode_list)
    # Initial with three nodes
    root_node = Bandelt_Node(sys.maxsize)
    node_1 = Bandelt_Node(1)
    node_neg_1 = Bandelt_Node(-1)
    node_0 = Bandelt_Node(0)
    # Create links between initial three nodes
    root_node.left = node_neg_1
    node_neg_1.parent = root_node
    node_neg_1.left = node_0
    node_neg_1.right = node_1
    node_1.parent = node_neg_1
    node_0.parent = node_neg_1
    
    for i in range(1, BANDELT_NUM):
        added_node_val = -(i+1)
#         print('Node going to be added: ', added_node_val)
#         print('Need to find: ', Bandelt_Encode_list[i], ' node')

        added_node_neg = Bandelt_Node(added_node_val)
        added_node_pos = Bandelt_Node(-added_node_val)

        target_node = root_node.find_node(Bandelt_Encode_list[i])

        # If target node is the left child 
        if target_node.parent.left == target_node:
            target_node.parent.left = added_node_neg
            added_node_neg.parent = target_node.parent
            added_node_neg.left = target_node
            target_node.parent = added_node_neg

            added_node_neg.right = added_node_pos
            added_node_pos.parent = added_node_neg

        # If target node is the right child
        if target_node.parent.right == target_node:
            target_node.parent.right = added_node_neg
            added_node_neg.parent = target_node.parent
            added_node_neg.right = target_node
            target_node.parent = added_node_neg

            added_node_neg.left = added_node_pos
            added_node_pos.parent = added_node_neg
    copy_root_node = copy.deepcopy(root_node)
    post_order_traversal_num_2_name(copy_root_node, file_path, file_base_name, mapping_dic_dic_decode)
    decode_tree = post_order_traversal_newick_string(copy_root_node)
    decode_tree_newick = decode_tree + ';'
    return (root_node, decode_tree_newick)