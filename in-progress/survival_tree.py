import random
import math
import numpy as np
import random

def get_random_subset(n_target, n_total):
    n_all = [i for i in range(n_total)]
    random.shuffle(n_all)
    return n_all[:n_target]

class node():

    def __init__(self, x, event, time, n_features, is_leaf = False):
        self.x = x # input data, will be deleted when the seperation takes place
        self.event = event
        self.time = time
        self.n_features = n_features
        self.split_value = None #
        self.data_left = None # left data
        self.data_right = None # right data
        self.left_node = None
        self.right_node = None
        self.split_position = None # (a,b) a tuple for splitting position
        self.is_leaf = is_leaf # is the current node a leaf?
        self.label = None # if so what is the label for this node?


    def _split_data(self, x, event, time, r, c):
        split_value = x[r][c]
        left_x, right_x, left_event, right_event, left_time, right_time = [], [],[], [],[], []

        for i in range(len(x)):

            if x[i][c] <= split_value:
                left_x.append(x[i])
                left_event.append(event[i])
                left_time.append(time[i])
            else:
                right_x.append(x[i])
                right_event.append(event[i])
                right_time.append(time[i])
        left_data = [np.asarray(left_x),np.asarray(left_event),np.asarray(left_time)]
        right_data = [np.asarray(right_x),np.asarray(right_event),np.asarray(right_time)]
        return left_data, right_data, split_value

    def _get_log_rank_splitting(self, group_index):

        numerator = 0.
        denom = 0.
        Y_j = 0.
        Y_j_L = 0.
        d_j = 0.
        d_j_L = 0.

        for i in range(len(self.time)):

            Y_j += 1

            if group_index[i] == 1 & self.event[i] == 1:
                Y_j_L += 1
                d_j_L += 1
                numerator += d_j_L - Y_j_L * (d_j / Y_j)
                if Y_j == 1:
                    denom += (Y_j_L / Y_j) * (1 - Y_j_L/Y_j) * ((Y_j - d_j)/(Y_j - .9))
                else:
                    denom += (Y_j_L / Y_j) * (1 - Y_j_L / Y_j) * ((Y_j - d_j) / (Y_j - 1))
            elif group_index[i] == 1 & self.event[i] != 1:
                Y_j_L += 1
            elif group_index[i] != 1 & self.event[i] == 1:
                d_j += 1
                numerator += d_j_L - Y_j_L * (d_j / Y_j)
                if Y_j == 1:
                    denom += (Y_j_L / Y_j) * (1 - Y_j_L/Y_j) * ((Y_j - d_j)/(Y_j - .9))
                else:
                    denom += (Y_j_L / Y_j) * (1 - Y_j_L / Y_j) * ((Y_j - d_j) / (Y_j - 1))
        return numerator / math.sqrt(denom)


    def _find_split(self, x, r, c):
        split_value = x[r][c]

        group_index = []
        for i in range(len(x)):
            if x[i][c] <= split_value:

                group_index.append(1)
            else:

                group_index.append(0)
        return group_index

    def _get_split_point(self,min_size,depth,max_depth):
        if len(self.x) < min_size:
            self._get_leaf_label()
            return
        n_total_features = self.x.shape[1]
        features = get_random_subset(self.n_features, n_total_features)
        r, c, log_rank = None, None, -999999.
        for row_index in range(len(self.time)):
            for feature in features:
                group_index = self._find_split(self.x, row_index, feature)
                lr_cur = self._get_log_rank_splitting(group_index)

                if lr_cur > log_rank:

                    self.data_left, self.data_right,self.split_value = self._split_data(self.x, self.event, self.time, row_index, feature)

                    self.split_position = (row_index, feature)

        if len(self.data_left[0]) == 0 or len(self.data_right[0]) == 0 or depth >= max_depth:

            self._get_leaf_label()

        self.x = None

    def _get_leaf_label(self):
        self.is_leaf = True
        self.chf, self.surv_func,self.func_ti = self._get_chf_surv()

    def _get_chf_surv(self):
        chf = []
        surv = []
        ti = []
        Y_j_h = 0.
        d_j_h = 0.
        surv_val = 0.
        for i in range(len(self.x)):
            Y_j_h += 1
            if self.event[i] == 1:
                d_j_h += 1
                ti.append(self.time[i])
                d_Y = d_j_h / Y_j_h
                chf.append(d_Y)

                surv_curr = np.log((1.001 - d_Y))

                surv_val += surv_curr
                surv.append(np.exp(surv_val))
        return chf, surv, ti

class survival_tree():

    def __init__(self,  max_depth, min_size, n_features, max_leaf_node = False):
        self.max_depth = max_depth
        self.min_size = min_size
        self.n_features = n_features
        self.leaf_node_total = 0
        self.max_leaf_node = max_leaf_node
        self.grown_tree = None

    def build_tree(self, cur_level_data, depth = 0):
        try:
            X, event, time = cur_level_data[0].values, cur_level_data[1].values, cur_level_data[2].values
        except:
            X, event, time = cur_level_data[0], cur_level_data[1], cur_level_data[2]


        self.order = np.argsort(-time, kind="quicksort")
        self.x = X[self.order, :]
        self.event = event[self.order]
        self.time = time[self.order]

        if self.max_leaf_node == False:
            pass

        else:
            if self.leaf_node_total >= self.max_leaf_node:
                return

        n = node(X, event, time ,self.n_features)
        n._get_split_point(self.min_size,depth,self.max_depth)
        left_group = n.data_left
        right_group = n.data_right
        print(left_group[0].shape, right_group[0].shape)
        n.data_left, n.data_right = None, None
        if n.is_leaf == False:
            n.left_node = self.build_tree(left_group, depth + 1)
            n.right_node = self.build_tree(right_group, depth + 1)
        else:
            self.leaf_node_total += 1
        return n

    def fit(self, cur_level_data):
        result = self.build_tree(cur_level_data)
        self.grown_tree = result
        print("Done")

    def predict_with_single_tree(self, init_tree, row):

        if init_tree.label is not None:
            return init_tree.label
        r,c = init_tree.split_position
        if row[c] <= init_tree.split_value:
            return self.predict_with_single_tree(init_tree.left_node, row)
        else:
            return self.predict_with_single_tree(init_tree.right_node, row)

    def predict(self, new_data):
        predictions = []
        for row in new_data:
            prediction = self.predict_with_single_tree(self.grown_tree, row)
            predictions.append(prediction)
        return predictions

if __name__ == '__main__':

    import pandas as pd
    whas = pd.read_csv("whas_new.csv")
    X = whas.iloc[:, 2:15]




    event = whas.iloc[:, [-7]]
    time = whas.iloc[:, [-5]]
    data = [X,event,time]


    tree = survival_tree(10,3,3)
    tree.fit(data)
    print(tree.grown_tree.left_node.right_node.is_leaf)
    """
    predictions = t.predict(data)
    print(predictions)
    """