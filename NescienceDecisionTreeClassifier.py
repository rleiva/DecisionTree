"""

Binary trees for classification based on the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   1.3 (Jun 2019)

TODO:
    - Adapt to coding standards of scikit-learn
       * validate with check_estimator
       * implement get_params and set_params methods
       * use project_template
       * implement unit tests
       * ...
    - Work in parallel
    - Provide support to sample weights
    - Provide support for partial fits
    - Allow categorical features
    - Extend to regression trees
    - Allow missing data

"""

import numpy  as np
import pandas as pd

import zlib

from sklearn.base import BaseEstimator, ClassifierMixin

class NescienceDecisionTreeClassifier(BaseEstimator, ClassifierMixin):
    """
    The following internal attributes will be used
    
      * root (dict)        - the computed tree
      * X (np.array)       - explanatory features
      * y (np.array)       - target values
      * classes_           - the classification classes labels
      * n_classes          - number of classification classes
      * nodesList (list)   - list of candidate nodes to growth
      * nescience (float)  - the best computed nescience so far
      * miscode (np.array) - global miscoding of each feature
      * code (np.array)    - optimal length of encoding classes
      * lcd (float)        - length of the target varible encoded
      * verbose (bool)     - print additional information
      * redundancy (float) - last redundancy computed
      * tolerance (float)  - allowed tolerance for redudancy

    Each node of the tree is a dictionary with the following format:
    
        * feature   - index of the column of the feature
        * value     - value for the split (feature < value)
        * left      - left branch (dict with same structure)
                      None if it is a leaf node
        * _left     - cache for left branch
        * right     - right branch (dict with same structure)
                      None if it is a leaf node
        * _right    - cache for right branch
        * lindices  - indices of current rows for left branch 
                      None if it is an intermediate node
        * rindices  - indices of current rows for right branch
                      None if it is an intermediate node
        * lforecast - forecasted value for the letf side
        * rforecast - forecasted value for the right side
        
    """
    
    def __init__(self, verbose=False):
        """
        Initialization of the tree

          * verbose: Boolean. If true, prints out additional information
        """
        
        self.verbose      = verbose
        
        
    def fit(self, X, y):
        """
        Fit a model (a tree) given a dataset
    
        The input dataset has to be in the following format:
    
           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
       
        Return the fitted model
        """
        
        # TODO: check the input parameters and init arguments

        self.X = np.array(X)
        
        # Transform categorial values into numbers
        self.classes_, self.y = np.unique(y, return_inverse=True)
        self.n_classes = self.classes_.shape[0]

        self.redundancy = 1      # Initial redundancy
        self.tolerance  = 0.5

        indices = np.arange(self.X.shape[0]) # Start with the full dataset

        # Compute the optimal code lengths for the response values
        self.code = self._codelength(indices)
        self.lcd  = np.sum(self.code[self.y])
                
        # Compute the contribution of each feature to miscoding         
        self.miscod = self._featuremiscod(indices)
        self.norm_miscod = 1 - np.array(self.miscod)
        total = np.sum(self.norm_miscod)
        self.norm_miscod = self.norm_miscod / total
        
        # Create the initial node
        self.root = self._create_node(indices)
        self.nescience  = self._nescience()
        self.tolerance  = abs(self.redundancy - self._redundancy()) / 2
        self.redundancy = self._redundancy()
        
        # List of candidate nodes to growth        
        self.nodesList = list()
        self.nodesList.append(self.root)
            
        if self.verbose:
            print("Miscoding: ", self._miscoding(), "Inaccuracy: ", self._inaccuracy(), "Redundancy: ", self._redundancy(), "Nescience: ", self._nescience())            

        # Build the tree
        self._fit()

        # Print out the best nescience achieved
        if self.verbose:
            print("Final nescience: " + str(self._nescience()))
            print(self._tree2str())

        return self


    """
    Fit the given dataset to a binary decision tree
    """
    def _fit(self):
                           
        # Meanwhile there are more nodes to growth
        while (self.nodesList):
            
            # Find the best node to develop
                        
            best_nsc  = 10e6    # A large value
            best_red  = 1
            best_node = 0
            best_side = ""
            
            for i in range(len(self.nodesList)):
                            
                # Get current node
                node = self.nodesList[i]
            
                # Try to create a left node
                #  - if empty
                #  - has more than one category
                
                if node['left'] == None  and len(np.unique(node['lindices'])) > 1:
                    
                    # Create the node, or get it from the cache
                    if node['_left'] == None:
                        node['left']  = self._create_node(node['lindices'])
                        node['_left'] = node['left']
                    else:
                        node['left']  = node['_left']
                                    
                    # Check if the node was created
                    if node['left'] != None:
                        
                        nsc = self._nescience()
                                                
                        # Save data if nescience has been reduced                        
                        if nsc < best_nsc:                                
                            best_nsc  = nsc
                            best_red  = self._redundancy()
                            best_node = i
                            best_side = "left"
                            # TODO: Use cache node instead
                            left = node['left']
                            
                        # And remove the node
                        node['left'] = None

                # Try to create a right node
                #  - if empty
                #  - has more than one category
                
                if node['right'] == None and len(np.unique(node['rindices'])) > 1:
                                
                    # Create the node, or get it from the cache
                    if node['_right'] == None:
                        node['right']  = self._create_node(node['rindices'])
                        node['_right'] = node['right']
                    else:                       
                        node['right']  = node['_right']
                
                    # Check if the node was created
                    if node['right'] != None:
                        
                        nsc = self._nescience()
                                                                                                                        
                        # Save data if nescience has been reduced                        
                        if nsc < best_nsc:                                
                            best_nsc  = nsc
                            best_red  = self._redundancy()
                            best_node = i
                            best_side = "right"
                            # TODO: Use cached node instead
                            right = node['right']
                            
                        # And remove the node
                        node['right'] = None
                                                    
            # -> end for
                                    
            # Stop the search if we failed to reduce the nescience
            if best_nsc >= self.nescience:
                break

            # Update the best nescience and the tolerance
            
            new_tolerance = abs(self.redundancy - best_red) / 2
            if new_tolerance < self.tolerance:
                self.tolerance = new_tolerance
                
            self.redunancy = best_red
            self.nescience = best_nsc

            # Add the best node found
            
            node = self.nodesList[best_node]

            if best_side == "left":
                node['left'] = left
                self.nodesList.append(node['left'])
                # Save space
                node['lindices'] = None
                node['_left']    = None
            else:
                node['right'] = right
                self.nodesList.append(node['right'])
                # Save space
                node['rindices'] = None
                node['_right']   = None
                
            # Delete the node from the list if it has been fully developed
            if node['left'] != None and node['right'] != None:
                del self.nodesList[best_node]
                            
            if self.verbose:
                print("Miscoding: ", self._miscoding(), "Inaccuracy: ", self._inaccuracy(), "Redundancy: ", self._redundancy(), "Nescience: ", self._nescience())
                
        # -> end while

        # There are no more nodes to growth, clean up the final tree
        self._cleanup(self.root, None, None)
        
        return
            

    def predict(self, X):
        """
        Predict class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
    
        Return a list of classes predicted
        """
        
        # TODO: Check that we have a model trained
        
        y = list()
        
        # For each entry in the dataset
        for i in np.arange(len(X)):
            y.append(self.classes_[self._forecast(self.root, X[i])])
                
        return y


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
          * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
      
        Return an array of probabilities. The order of the list match the order
        the internal attribute classes_
        """
        
        # TODO: Check that we have a model trained
        
        proba = list()
        
        # For each entry in the dataset
        for i in np.arange(len(X)):
            my_list = self._proba(self.root, X[i])
            proba.append(my_list)
            
        return np.array(proba)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

           * X = list([[x11, x12, x13, ...], ..., [xn1, xn2, ..., xnm]])
           * y = list([y1, ..., yn])
    
        Return one minus the mean error
        """
        
        # TODO: Check that we have a model trained
        
        error = 0

        # For each entry in the dataset
        for i in np.arange(len(X)):
            
            est = self.classes_[self._forecast(self.root, X[i])]
                    
            if est != y[i]:
                error = error + 1
        
        score = 1 - error / len(X)
        
        return score


    """
    Compute the contribution of each feature to miscoding given a subset
    
      * indices: the indices of the subset considered
      
    Return a list of miscoding values
    """
    def _featuremiscod(self, indices):
         
        miscod = list()
        
        Resp = self.y[indices]
        unique, count_y = np.unique(Resp, return_counts=True)
        ldm_y = np.sum(count_y  * ( - np.log2(count_y  / len(self.y[indices]))))

        for i in np.arange(self.X.shape[1]):

            # Discretize the feature
            if len(np.unique(self.X[indices][:,i])) == 1:
                # Do not split if all the points belong to the same category
                Pred = np.zeros(len(self.y[indices]))
            else:
                nbins = int(np.sqrt(len(self.y[indices])))
                tmp   = pd.qcut(self.X[indices][:,i], q=nbins, duplicates='drop')
                Pred  = list(pd.Series(tmp).cat.codes)
            
            Join =  list(zip(Pred, Resp))
            
            unique, count_X  = np.unique(Pred, return_counts=True)
            unique, count_Xy = np.unique(Join, return_counts=True, axis=0)
            
            tot = self.X[indices].shape[0]

            ldm_X   = np.sum(count_X  * ( - np.log2(count_X  / tot)))
            ldm_Xy  = np.sum(count_Xy * ( - np.log2(count_Xy / tot)))
            
            mscd = ( ldm_Xy - min(ldm_X, ldm_y) ) / max(ldm_X, ldm_y)

            miscod.append(mscd)
                
        return np.array(miscod)


    """"
    Compute the minimum length of a code to encode the response values y
    """
    def _codelength(self, indices):
                
        unique, count = np.unique(self.y[indices], return_counts=True)
        code  = np.zeros(self.n_classes)
        
        for i in np.arange(len(unique)):
            code[unique[i]] = - np.log2( count[i] / len(self.y[indices]) )
            
        return code


    """
    Recursively compute a forecast given a list of values
    
       * node - the current node being evaluated
       * values - a list of values used for forecasting
    
    Return the forecasted value (int)
    """
    def _forecast(self, node, values):
        
        index = node['feature']
        value = node['value']

        if values[index] < value:
            if node['left'] != None:
                y = self._forecast(node['left'], values)
            else:
                return node['lforecast']
        else:
            if node['right'] != None:
                y = self._forecast(node['right'], values)
            else:
                return node['rforecast']

        return y
    
    
    """
    Helper function to compute the probability of a forecasted value
    """
    def _proba(self, node, values):
        
        index = node['feature']
        value = node['value']

        if values[index] < value:
            if node['left'] != None:
                prob_list = self._proba(node['left'], values)
                return prob_list

            else:
                indices = self.y[node['lindices']]
        else:
            if node['right'] != None:
                prob_list = self._proba(node['right'], values)
                return prob_list

            else:
                indices = self.y[node['rindices']]

        prob_list = list()
        length = len(indices)
        
        for i in np.arange(self.n_classes):
            prob_list.append(np.sum(self.y[indices] == i) / length)
            
        return prob_list


    """
    Compute the forecast values for a node
        
     * lindices - indices of the left dataset
     * rindices - indices of the right dataset
     
    Return the forecasted values
    """
    def _compute_forecast(self, lindices, rindices):

        # TODO: Use numpy instead of lists        
        # unique, counts = numpy.unique(a, return_counts=True)
        # dict(zip(unique, counts))
        
        lly = list(self.y[lindices])
        lry = list(self.y[rindices])
                  
        lforecast = max(set(lly), key=lly.count)
        rforecast = max(set(lry), key=lry.count)
                        
        return lforecast, rforecast


    """
    Helper function to recursively compute the miscoding
    """
    def _miscodingcount(self, node):
                        
        self.var_in_use[node['feature']] = 1
        
        # Process left branch
        if node['left'] != None:
            self._miscodingcount(node['left'])
    
        # Process right branch    
        if node['right'] != None:
            self._miscodingcount(node['right'])
                
        return

    
    """
    Compute the global miscoding of the dataset used by the current tree
      
    Return the miscoding (float)
    """
    def _miscoding(self):
        
        # TODO: Use a global accounting
            
        self.var_in_use = np.zeros(self.X.shape[1])

        self._miscodingcount(self.root)

        miscoding = np.dot(self.var_in_use, self.norm_miscod)
        miscoding = 1 - miscoding
                            
        return miscoding


    """
    Compute global inaccuracy of the current tree

    Return the inaccuracy (float)
    """
    def _inaccuracy(self):
                        
        # TODO: Use a global accounting
        
        # Compute the list of errors
        errors = list()
        for i in np.arange(self.X.shape[0]):
            pred = self._forecast(self.root, self.X[i])
            if pred != self.y[i]:
                errors.append(self.y[i])

        # Compute the length of the encoding of the error
        ldm = np.sum(self.code[errors])
                        
        # Inaccuracy = l(d/m) / l(d)
        inaccuracy = ldm / self.lcd
        
        return inaccuracy


    """
    Compute the redundancy of the current tree model
    
    Return the redundancy (float)
    """
    def _redundancy(self):
    
        # Compute the model string and its compressed version
        model      = self._tree2str().encode()
        compressed = zlib.compress(model, level=9)
        km         = len(compressed)
        
        # Check if the model is too small to compress
        if km > len(model):
            # km = len(model)
            redundancy = 1 - 3/4    # Experimental values for zlib
        else:
            # redundancy = 1 - l(m*) / l(m)
            redundancy = 1 - km / len(model)
            
        return redundancy


    """
    Compute the nescience of a tree
              
    Return the nescience (float)
    """
    def _nescience(self):

        miscoding  = self._miscoding()
        redundancy = self._redundancy()
        inaccuracy = self._inaccuracy()

        if redundancy == 0:
            # Avoid dividing by zero
            # TODO: shall I use "inaccuracy = np.finfo(np.float32).tiny" instead?
            redundancy = 10e-6

        # TODO: Provide a theoretical interpretation of this decision
        if redundancy < inaccuracy + self.tolerance:
            # The model is still too small to compute the nescience
            # use innacuracy instead
            redundancy = 1
    
        if inaccuracy == 0:
            # Avoid dividing by zero
            inaccuracy = 10e-6

        if miscoding == 0:
            # Avoid dividing by zero
            miscoding = 10e-6

        # Compute the nescience using an harmonic mean
        nescience = 3 / ( (1/miscoding) + (1/inaccuracy) + (1/redundancy))
            
        return nescience


    """
    Split a dataset based on an attribute and an attribute value
    
      * attribute   - column number of the attribute
      * value       - value of the attribute for the split
      * indices     - array with the indices of the rows to split
    
    Return
    
      * lindices - numpy array with those indices smaller than
      * rindices - numpy array with those indices greater or equal than
    """
    def _split_data(self, attribute, value, indices):

        lindex = np.where(self.X[:,attribute] < value)
        rindex = np.where(self.X[:,attribute] >= value)

        lindices = np.intersect1d(indices, lindex)
        rindices = np.intersect1d(indices, rindex)
                
        return lindices, rindices


    """
    Create a new tree node based on the best split point for the given dataset
    Splitting criteria is a reduced minimum nescience principle (miscoding and
    inaccuracy)
    
      * indices - indices of the rows to split
    
    Return a new node (dict)
    """   
    def _create_node(self, indices):

        best_nescience  = 10e6    # A large value
        best_feature    = None
        best_value      = None
        best_lindices   = None
        best_rindices   = None
        best_lforecast  = None
        best_rforecast  = None
        
        # Do not split if all the points belong to the same category
        if len(np.unique(self.y[indices])) == 1:
            return None

        # Use a local splitting criteria
        code      = self._codelength(indices)
        lcd       = np.sum(code[self.y[indices]])
        miscoding = self._featuremiscod(indices)
            
        #
        # Search for the best spliting feature based on the local MNP
        # Surfeit is not used since it is locally constant
        #
                
        for feature in np.arange(self.X.shape[1]):
                                    
            # Some datasets repeat values many times
            values = np.unique(self.X[indices][:,feature])
                            
            # We cannot use this feature if it has no possible splitting points
            if len(values) <= 1:
                continue

            # We need the values sorted to compute middle points
            values.sort()
            
            #
            # Search for the best splitting value
            #
        
            for i in np.arange(1, len(values)):
            
                lindices, rindices = self._split_data(feature, values[i], indices)
            
                # If one side is empty we cannot split using this value
                if len(lindices) == 0 or len(rindices) == 0:
                    continue

                # Compute the size of errors
            
                lforecast, rforecast = self._compute_forecast(lindices, rindices)        
                        
                lerrors = np.where(self.y[lindices] != lforecast)
                rerrors = np.where(self.y[rindices] != rforecast)
                errors  = np.concatenate((lerrors, rerrors), axis=None)
                      
                # ldm = np.sum(code[self.y[errors]])
                ldm = np.sum(code[self.y[errors]])
                                        
                # Inaccuracy = l(d/m) / l(d)
                inaccuracy = ldm / lcd
                
                # Compute the nescience
                
                if inaccuracy == 0:
                    # Avoid dividing by zero
                    inaccuracy = 10e-6
                    
                if miscoding[feature] == 0:
                    # Avoid dividing by zero
                    misc = 10e-6
                else:
                    misc = miscoding[feature]    

                # Nescience is the product of both quantities
                nescience = misc * inaccuracy
                                   
                # Check if we have found a better spliting point
                if nescience < best_nescience:
                    best_feature   = feature
                    best_nescience = nescience
                    best_value     = (values[i-1] + values[i]) / 2
                    best_lindices  = lindices
                    best_rindices  = rindices
                    best_lforecast = lforecast
                    best_rforecast = rforecast
                
        # Check if we have found a good split
        if best_feature == None:
            return None

        # Create the new node
        my_dict = {'feature':   best_feature,
                   'value':     best_value,
                   'left':      None,
                   'right':     None,
                   '_left':     None,
                   '_right':    None,                   
                   'lindices':  best_lindices,
                   'rindices':  best_rindices,
                   'lforecast': best_lforecast,
                   'rforecast': best_rforecast}
        
        return my_dict        


    """
    Helper function to recursively compute the head of the tree
    """
    def _head2str(self, node):

        myset = set()

        # Print the feature to take at this level
        myset.add('X%d' % (node['feature']+1))

        # Process left branch
        if node['left'] != None:
            myset = myset.union(self._head2str(node['left']))
    
        # Process right branch    
        if node['right'] != None:
            myset = myset.union(self._head2str(node['right']))
                
        return myset


    """
    Helper function to recursively compute the body of the tree
    """
    def _body2str(self, node, depth):

        string = ""

        # Print the decision to take at this level
        string = string + '%sif X%d < %.3f:\n' % (' '*depth*4, (node['feature']+1), node['value'])

        # Process left branch
        if node['left'] == None:
            string = string + '%sreturn %s\n' % (' '*(depth+1)*4, self.classes_[node['lforecast']])
        else:
            string = string + self._body2str(node['left'],  depth+1)
    
        # Process right branch
        string = string + '%selse:\n' % (' '*depth*4)
    
        if node['right'] == None:
            string = string + '%sreturn %s\n' % (' '*(depth+1)*4, self.classes_[node['rforecast']])
        else:
            string = string + self._body2str(node['right'], depth+1)
                
        return string

    
    """
    Convert a tree into a string
    
    Convert a decision tree into a string using an austere representation
    Intended to compute the nescience of the tree
    although it can be used for visualization purposes as well
        
    Return a string with a representation of the tree
    """
    def _tree2str(self):
    
        # Compute the tree header
        string = "def tree" + str(self._head2str(self.root)) + ":\n"

        # Compute the tree body
        string = string + self._body2str(self.root, 1)

        return string


    """
    Helper function to recursively compute the number of nodes
    """
    def _bodycount(self, node):

        nodes = 1
        
        # Process left branch    
        if node['left'] == None:
            nodes = nodes + 1
        else:
            nodes = nodes + self._bodycount(node['left'])
    
        # Process right branch    
        if node['right'] == None:
            nodes = nodes + 1
        else:
            nodes = nodes + self._bodycount(node['right'])
                
        return nodes

    
    """
    Compute the total number of nodes of the tree
    
    Return total number of nodes (int)
    """
    def _nodecount(self):
    
        return(self._bodycount(self.root))
    
    
    """
    Helper function to recursively compute the depth of a tree
    """
    def _bodydepth(self, node):
        
        ldepth = 0
        rdepth = 0
        
        if node['left'] != None:
            ldepth = self._bodydepth(node['left'])
        
        if node['right'] != None:
            rdepth = self._bodydepth(node['right'])

        if ldepth > rdepth:
            return ldepth + 1
        else:
            return rdepth + 1


    """
    Compute the depth of the current tree (length of the longest branch)
    
    Return depth of the tree (int) 
    """
    def _maxdepth(self):
    
        return(self._bodydepth(self.root))


    """
    Clean up a tree removing redundant nodes, that is, nodes where both
    sides forecast the same value
    """
    def _cleanup(self, node, parent, side):
        
        # TODO: We should repeat this process recursively
        # meanwhile nodes have been removed
        
        # Check if it is a terminal node
        if node['left'] == None and node['right'] == None:
            # If both sides forecast the same value, clean it up
            if node['lforecast'] == node['rforecast']:
                if parent != None:    # Avoid to clean up the root node
                    if side == 'left':
                        parent['left']     = None
                        parent['lindices'] = np.concatenate((node['lindices'], node['rindices']))
                    else:
                        parent['right']    = None
                        parent['rindices'] = np.concatenate((node['lindices'], node['rindices']))
                    
        else:
            if node['left'] != None:
                self._cleanup(node['left'], node, "left")
        
            if node['right'] != None:
                self._cleanup(node['right'], node, "right")

