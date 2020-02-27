import sys, os
import os

class Bandelt_Node:
    def __init__(self, data):
        self.parent = None
        self.left = None
        self.right = None
        self.data = data
        
    def find_node(self, val):
        if self.data == val:
            return self
        else:
            if (self.left == None) and (self.right == None):
                return None
            if self.left != None:
                find_left = self.left.find_node(val)
                if find_left != None:
                    return find_left
            if self.right != None:
                find_right = self.right.find_node(val)
                if find_right != None:
                    return find_right
            return None
        
    def print_details(self):
        parent_data = self.parent.data if (self.parent != None) else self.parent
        left_data = self.left.data if (self.left != None) else self.left
        right_data = self.right.data if (self.right != None) else self.right
        print("Current Value: ", self.data, "; Parent: ", parent_data, "; left: ", left_data, "; right: ", right_data)
        
    def print_subtree(self, indent_num = 1):
        if indent_num == 1:
            print(self.data)
        if self.left != None:
            print('___'*indent_num, self.left.data)
            self.left.print_subtree(indent_num + 1)
        if self.right != None:
            print('___'*indent_num, self.right.data)
            self.right.print_subtree(indent_num + 1)
            
    def compare_subtree(self, compared_root_node):
        if self.data != compared_root_node.data:
            return False
        
        self_left_data = self.left.data if (self.left != None) else None
        self_right_data = self.right.data if (self.right != None) else None

        compared_root_node_left_data = compared_root_node.left.data if (compared_root_node.left != None) else None
        compared_root_node_right_data = compared_root_node.right.data if (compared_root_node.right != None) else None

        
        if (self_left_data in [compared_root_node_left_data, compared_root_node_right_data]) and (self_right_data in [compared_root_node_left_data, compared_root_node_right_data]):
            if (self_left_data == compared_root_node_left_data):
                if self.left != None:
                    compare_ans_left = self.left.compare_subtree(compared_root_node.left)
                    if not compare_ans_left:
                        return False
                if self.right != None:
                    compare_ans_right = self.right.compare_subtree(compared_root_node.right)
                    if not compare_ans_right:
                        return False
                
            elif (self_left_data == compared_root_node_right_data):
                if self.left != None:
                    compare_ans_left = self.left.compare_subtree(compared_root_node.right)
                    if not compare_ans_left:
                        return False
                if self.right != None:
                    compare_ans_right = self.right.compare_subtree(compared_root_node.left)
                    if not compare_ans_right:
                        return False
        else:
            return False
        return True