"""
Build a binary tree for classification and regression problems
based on the minimum nescience principle

@author:    Rafael Garcia Leiva
@mail:      rgarcialeiva@gmail.com
@web:       http://www.mathematicsunknown.com/
@copyright: GNU GPLv3
@version:   1.6 (Jun 2021)
"""

import numpy  as np
import pandas as pd

import zlib
import warnings
import copy
import math

from joblib import Parallel, delayed
from collections import defaultdict

from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing    import LabelEncoder
from sklearn.preprocessing    import KBinsDiscretizer

from sklearn.base import BaseEstimator 
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin

# Constants
CONSTANT_FEATURE    = -1    # Attribute is constant
NO_SPLIT_FOUND      = -2    # No suitable split has been found for attribute
CAN_BE_DEVELOPED    = -3    # Node cannot be futher develped
CANNOT_BE_DEVELOPED = -4    # Node can be developed

DEBUG = False    # Print addtional information

class NescienceDecisionTree(BaseEstimator):
    """
    The following class variables will be used

      * X_ (np.array)        - explanatory attributes
      * y_ (np.array)        - target values
      * root_ (dict)         - root of the computed tree      
      * nescience_ (float)   - the best computed nescience

      * isRegression (bool)  - if the target problem is "regression" or "classification" 
      * isNumeric (np.array) - if the attributes are numeric or not
      * nodesList (list)     - list of candidate nodes to growth
      * miscoding (np.array) - global miscoding of each attribute
      * discretizer (KBins)  - KBinsDiscretizer for continuous targets
      * code (np.array)      - optimal length of encoding classes or intervals
      * lengh_y (float)      - length of the target varible encoded
      * verbose (bool)       - print additional information

    For classification problems

      * classes_             - the classification classes labels
      * n_classes            - number of classification classes

    Each node of the tree is a dictionary with the following structure:

        * attribute  - index of the column of the attribute
        * value      - value for the split
        * operator   - either '<' or '='
        * left       - left branch (dict with same structure)
                       CANNOT_BE_DEVELOPED if it is a leaf node
        * _left      - cache for left branch
        * right      - right branch (dict with same structure)
                       CANNOT_BE_DEVELOPED if it is a leaf node
        * _right     - cache for right branch
        * lindices   - indices of current rows for left branch 
                       None if it is an intermediate node
        * rindices   - indices of current rows for right branch
                       None if it is an intermediate node
        * lforecast  - forecasted value for the letf side
        * rforecast  - forecasted value for the right side
    """
    
    def __init__(self, mode="regression", cleanup=True, partial_miscoding=True, early_stop=True, verbose=False, n_jobs=1):
        """
        Initialization of the tree

          * mode (string)     : Type of target problem, "regression" or "classification"
          * cleanup(bool)     : If True, redundant leaf nodes are removed from final tree
          * n_jobs (int)      : Number of concurrent jobs
          * partial_miscoding : use partial miscoding instead of adjusted miscoding
          * early_stop        : stop the algorithm as soon as a good solution has been found
          * verbose (bool)    : If True, prints out additional information
        """
        
        valid_modes = ("regression", "classification")

        if mode not in valid_modes:
            raise ValueError("Valid options for 'mode' are {}. "
                             "Got vartype={!r} instead."
                             .format(valid_modes, mode))

        if mode == "regression":
            self.isRegression = True
        else:
            self.isRegression = False

        self.cleanup           = cleanup
        self.partial_miscoding = partial_miscoding
        self.early_stop        = early_stop
        self.n_jobs            = n_jobs
        self.verbose           = verbose
    
    
    def fit(self, X, y):
        """
        Fit a tree given a dataset
    
        X : array-like, shape (n_samples, n_attributes)
            Sample vectors from which to train the tree.
            array-like, numpy or pandas array in case of numerical attributes
            pandas array in case of mixed (numeric and categorical) attributes
            
        y : array-like, shape (n_samples)
            The target values as numbers or strings.

        Return the fitted model
        """

        self.X_ = np.array(X)
        self.y_ = np.array(y)

        if isinstance(X, pd.DataFrame):
            self.X_isNumeric = [np.issubdtype(my_type, np.number) for my_type in X.dtypes]
        else:
            self.X_isNumeric = [True] * self.X_.shape[1]

        if self.isRegression:
            # Discretize target values
            discretizer = self._discretizer(self.y_.reshape(-1, 1))
            self.y_ = discretizer.transform(self.y_.reshape(-1, 1))[:,0].astype(dtype=int)
            self.classes_ = np.unique(discretizer.inverse_transform(self.y_.reshape(-1, 1)))
            classes, self.y_, count = np.unique(self.y_, return_inverse=True, return_counts=True)
        else:
            # No discretization needed
            self.discretizer = None
            # Transform categorial values into numbers
            self.classes_, self.y_, count = np.unique(y, return_inverse=True, return_counts=True)

        self.n_classes = len(self.classes_)

        # Compute the optimal code lengths for the response values
        self.length_y = 0
        for i in np.arange(self.n_classes):
            self.length_y = self.length_y - np.log2( count[i] / len(self.y_) ) * count[i]

        # Compute the contribution of each attribute to miscoding
        regular  = self._attributes_miscoding()
        adjusted = 1 - regular
        adjusted = adjusted / np.sum(adjusted)
        if self.partial_miscoding:
            self.miscoding = adjusted - regular / np.sum(regular)
        else:
            self.miscoding = adjusted

        # Create the initial node
        indices        = np.arange(self.X_.shape[0])  # Start with the full dataset
        constants      = np.array([False] * self.X_.shape[1]) # Attribute known to be constant
        self.root_     = self._create_node(indices, constants)
        self.best_tree = copy.deepcopy(self.root_)

        # Compute y_hat
        self.y_hat_ = np.zeros(len(self.y_))
        self.y_hat_[self.root_['lindices']] = self.root_['lforecast']
        self.y_hat_[self.root_['rindices']] = self.root_['rforecast']
        
        # Initial variables in use
        self.var_in_use_ = np.zeros(self.X_.shape[1])
        self.var_in_use_[self.root_['attribute']] = 1

        # Intial nescience
        self.nescience_ = self._nescience()

        # List of candidate nodes to growth        
        self.nodesList = list()
        self.nodesList.append(self.root_)
        
        if DEBUG:
            print("Miscoding: ",  self._miscoding(), 
                  "Inaccuracy: ", self._inaccuracy(),
                  "Surfeit: ",    self._surfeit(),
                  "Nescience: ",  self._nescience())

        if self.verbose:
            print("Best Nescience:",   self.nescience_)           

        # Build the tree
        self._fit()

        # Print out the best nescience achieved
        if self.verbose:
            print(self._tree2str())

        return self


    """
    Fit the given dataset to a binary decision tree
    """
    def _fit(self):
                           
        # Meanwhile there are more nodes to growth
        while (self.nodesList):
            
            # Find the best node to develop
                        
            best_nsc   = 10e6    # A large value
            best_node  = -1
            best_side  = ""
            
            for i in range(len(self.nodesList)):
                            
                # Get current node
                node = self.nodesList[i]

                # Control if the node can be developed
                left_side  = False
                right_side = False
            
                # Try to create a left node if empty
                if node['left'] == CAN_BE_DEVELOPED:
                    
                    # Create the node, or get it from the cache
                    if node['_left'] == CAN_BE_DEVELOPED:
                        node['left']  = self._create_node(node['lindices'], node['constants'])
                    else:
                        node['left']  = node['_left']
                                    
                    # Check if the node was created
                    if node['left'] != CANNOT_BE_DEVELOPED:

                        # Left side can be developed
                        left_side = True

                        # Save node in cache
                        node['_left'] = node['left']

                        # Compute y_hat
                        y_hat = self.y_hat_.copy()
                        y_hat[node['left']['lindices']] = node['left']['lforecast']
                        y_hat[node['left']['rindices']] = node['left']['rforecast']
                        
                        nsc = self._nescience(y_hat=y_hat, attr=node['left']['attribute'])
                                                
                        # Save data if nescience has been reduced                        
                        if nsc < best_nsc:                                
                            best_nsc   = nsc
                            best_node  = i
                            best_side  = "left"
                            left       = node['left']
                            
                        # And remove the node
                        node['left'] = CAN_BE_DEVELOPED

                # Try to create a right node if empty
                if node['right'] == CAN_BE_DEVELOPED:
                                
                    # Create the node, or get it from the cache
                    if node['_right'] == CAN_BE_DEVELOPED:
                        node['right']  = self._create_node(node['rindices'], node['constants'])
                    else:                       
                        node['right']  = node['_right']
                
                    # Check if the node was created
                    if node['right'] != CANNOT_BE_DEVELOPED:

                        # Right side can be developed
                        right_side = True

                        # Save node in cache
                        node['_right'] = node['right']

                        # Compute y_hat
                        y_hat = self.y_hat_.copy()
                        y_hat[node['right']['lindices']] = node['right']['lforecast']
                        y_hat[node['right']['rindices']] = node['right']['rforecast']
                        
                        nsc = self._nescience(y_hat=y_hat, attr=node['right']['attribute'])

                        # Save data if nescience has been reduced                        
                        if nsc < best_nsc:                                
                            best_nsc   = nsc
                            best_node  = i
                            best_side  = "right"
                            right      = node['right']
                            
                        # And remove the node
                        node['right'] = CAN_BE_DEVELOPED

                # Mark the node if it cannot be developed
                if (left_side == False) and (right_side == False):
                    self.nodesList[i] = None

            # -> end for

            if best_node != -1:

                # Add the best node found (if any)
            
                node = self.nodesList[best_node]

                if best_side == "left":
                    node['left'] = left
                    self.nodesList.append(node['left'])
                    # Update pre-computed values
                    self.y_hat_[node['left']['lindices']] = node['left']['lforecast']
                    self.y_hat_[node['left']['rindices']] = node['left']['rforecast']
                    self.var_in_use_[node['left']['attribute']] = 1
                    # Save space
                    node['lindices'] = None
                    node['_left']    = CANNOT_BE_DEVELOPED
                else:
                    node['right'] = right
                    self.nodesList.append(node['right'])
                    # Update y_hat
                    self.y_hat_[node['right']['lindices']] = node['right']['lforecast']
                    self.y_hat_[node['right']['rindices']] = node['right']['rforecast']
                    self.var_in_use_[node['right']['attribute']] = 1
                    # Save space
                    node['rindices'] = None
                    node['_right']   = CANNOT_BE_DEVELOPED

            # Clean up the list of nodes
            self.nodesList = [node for node in self.nodesList if node]

            # Update best tree
            if best_nsc < self.nescience_:
                # Update the best values
                self.nescience_ = best_nsc
                self.best_tree  = copy.deepcopy(self.root_)

                if self.verbose:
                    print("Best Nescience:",   self.nescience_)

            else:
                # Stop if the user only wants a good enougth solution
                if self.early_stop:
                    # Avoid the inestability of inaccuracy
                    if self._inaccuracy() < self._surfeit():
                        break
                            
            if DEBUG:
                print("Miscoding: ",  self._miscoding(), 
                      "Inaccuracy: ", self._inaccuracy(),
                      "Surfeit: ",    self._surfeit(),
                      "Nescience: ",  self._nescience())

        # -> end while

        self.root_ = self.best_tree
        self.pc_miscoding = self._miscoding()

        # There are no more nodes to growth, clean up the final tree
        if self.cleanup:
            flag = True
            while flag:
                flag = self._cleanup(self.root_, None, None)
        
        return
            

    def predict(self, X):
        """
        Predict class given a dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        A list with the classes or values predicted
        """
        
        check_is_fitted(self)
        
        X_ = np.array(X)
        y = list()
        
        # For each entry in the dataset
        for i in np.arange(len(X_)):
            y.append(self.classes_[self._forecast(self.root_, X_[i])])
                
        return y


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        An array of probabilities. The order of the list match the order
        the internal attribute classes_
        """
        
        check_is_fitted(self)
        
        X_ = np.array(X)
        proba = list()
        
        # For each entry in the dataset
        for i in np.arange(len(X_)):
            my_list = self._proba(self.root_, X_[i])
            proba.append(my_list)
            
        return np.array(proba)


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
        y : array-like, shape (n_samples)
    
        Returns
        -------
        Classification: One minus the mean error
        Regression: root mean squared error
        """
        
        check_is_fitted(self)

        X_ = np.array(X)
        y_ = np.array(y)

        if self.isRegression:
            y_mean = np.mean(y_)
        
        error = 0
        u = v = 0

        # For each entry in the dataset
        for i in np.arange(X_.shape[0]):
            
            y_hat = self.classes_[self._forecast(self.root_, X_[i])]
            
            if self.isRegression:
                u = u + (y_[i] - y_hat)**2
                v = v + (y_[i] - y_mean)**2
            else:
                if y_hat != y_[i]:
                    error = error + 1
        
        if self.isRegression:
            score = 1 - u / v
        else:
            score = 1 - error / X_.shape[0]
        
        return score


    """
    Count the number of occurences of a discretized 1d or 2d space
    for classification or regression problems
    
    Parameters
    ----------
    x1, x2: array-like, shape (n_samples)
    numeric1, numeric2: if the variable is numeric or not
    double: return count for x1 and joint x1 and x2
       
    Returns
    -------
    A vector with the frequencies of the unique values computed.
    """
    def _unique_count(self, x1, numeric1, x2=None, numeric2=None, double=False):

        # Process first variable

        if not numeric1:
            # Econde categorical values as numbers
            le = LabelEncoder()
            le.fit(x1)
            x1 = le.transform(x1)
        else:
            # Discretize variable
            x1 = self._discretize_vector(x1)

        # Process second variable

        if x2 is not None:

            if not numeric2:
                # Econde categorical values as numbers
                le = LabelEncoder()
                le.fit(x2)
                x2 = le.transform(x2)
            else:
                # Discretize variable
                x2 = self._discretize_vector(x2)

            # Apply Cantor's formula
            x3 = (x1 + x2) * (x1 + x2 + 1) / 2 + x2
            x3 = x3.astype(int)

        else:
        
            x3 = x1

        # Return count

        if double:

            y      = np.bincount(x1)
            ii     = np.nonzero(y)[0]
            count1 = y[ii]
            y      = np.bincount(x3)
            ii     = np.nonzero(y)[0]
            count2 = y[ii]

            return count1, count2

        else:

            y      = np.bincount(x3)
            ii     = np.nonzero(y)[0]
            count = y[ii]

            return count


    """
    Compute a discretizer for a continous variable using a "uniform" strategy
    
    Parameters
    ----------
    x  : array-like, shape (n_samples)
       
    Returns
    -------
    KBinsDiscritizer already fitted
    """
    def _discretizer(self, x):

        length = x.shape[0]

        # Optimal number of bins
        optimal_bins = int(np.cbrt(length))
    
        # Correct the number of bins if it is too small
        if optimal_bins <= 1:
            optimal_bins = 2
    
        # Repeat the process until we have data in all the intervals

        total_bins    = optimal_bins
        previous_bins = 0
        stop          = False

        while stop == False:

            # Avoid those annoying warnings
            with warnings.catch_warnings():
            
                warnings.simplefilter("ignore")

                est = KBinsDiscretizer(n_bins=total_bins, encode='ordinal', strategy="uniform")
                est.fit(x)
                tmp_x = est.transform(x.reshape(-1, 1))[:,0].astype(dtype=int)

            y = np.bincount(tmp_x)
            actual_bins = len(np.nonzero(y)[0])

            if previous_bins == actual_bins:
                # Nothing changed, better stop here
                stop = True

            if actual_bins < optimal_bins:
                # Too few intervals with data
                add_bins      = optimal_bins - actual_bins
                previous_bins = actual_bins
                total_bins    = total_bins + add_bins
            else:
                # All intervals have data
                stop = True

        return est


    """
    Discretize a continous variable using a "uniform" strategy
    
    Parameters
    ----------
    x  : array-like, shape (n_samples)
    dim: Number of dimensions of space
       
    Returns
    -------
    A new discretized vector of integers.
    """
    def _discretize_vector(self, x):

        est   = self._discretizer(x.reshape(-1, 1))
        new_x = est.transform(x.reshape(-1, 1))[:,0].astype(dtype=int)

        return new_x


    """
    Compute the length of a list of attributes (1d or 2d)
    and / or a target variable (classification or regression)
    using an optimal code using the frequencies of the categorical variables
    or a discretized version of the continuous variables
        
    Parameters
    ----------
    x1, x2: array-like, shape (n_samples)
    numeric1, numeric2: if the variable is numeric or not
    dourble: return the optimal length for x1 and the joint x1 and x2
       
    Returns
    -------
    Return the length of the encoded dataset (float)
    """
    def _optimal_code_length(self, x1, numeric1, x2=None, numeric2=None, double=False):

        if double:
            count1, count2 = self._unique_count(x1=x1, numeric1=numeric1, x2=x2, numeric2=numeric2, double=True)
            ldm1 = np.sum(count1 * ( - np.log2(count1 / len(x1) )))
            ldm2 = np.sum(count2 * ( - np.log2(count2 / len(x1) )))
            return ldm1, ldm2
        else:
            count = self._unique_count(x1=x1, numeric1=numeric1, x2=x2, numeric2=numeric2, double=False)
            ldm = np.sum(count * ( - np.log2(count / len(x1) )))
            return ldm


    """
    Compute the contribution of each attribute to miscoding
      
    Return a list of miscoding values
    """
    def _attributes_miscoding(self):
         
        miscoding = list()

        for i in np.arange(self.X_.shape[1]):
            
            ldm_X, ldm_Xy = self._optimal_code_length(x1=self.X_[:,i], numeric1=self.X_isNumeric[i],
                                                      x2=self.y_, numeric2=self.isRegression, double=True)
            mscd = ( ldm_Xy - min(ldm_X, self.length_y) ) / max(ldm_X, self.length_y)
            
            miscoding.append(mscd)
                
        miscoding = np.array(miscoding)
        
        return miscoding


    """
    Compute a forecast given a list of values
    
       * node - the current node being evaluated
       * values - a list of values used for forecasting
    
    Return the forecasted value (int)
    """
    def _forecast(self, node, values):

        curr_node = node

        while True:

            index    = curr_node['attribute']
            value    = curr_node['value']
            operator = curr_node['operator']

            if operator == '<':

                if values[index] < value:
                    if curr_node['left'] != CAN_BE_DEVELOPED and curr_node['left'] != CANNOT_BE_DEVELOPED:
                        curr_node = curr_node['left']
                    else:
                        return curr_node['lforecast']
                else:
                    if curr_node['right'] != CAN_BE_DEVELOPED and curr_node['right'] != CANNOT_BE_DEVELOPED:
                        curr_node = curr_node['right']
                    else:
                        return curr_node['rforecast']

            else:    # operator is '='

                if values[index] == value:
                    if curr_node['left'] != CAN_BE_DEVELOPED and curr_node['left'] != CANNOT_BE_DEVELOPED:
                        curr_node = curr_node['left']
                    else:
                        return node['lforecast']
                else:
                    if curr_node['right'] != CAN_BE_DEVELOPED and curr_node['right'] != CANNOT_BE_DEVELOPED:
                        curr_node = curr_node['right']
                    else:
                        return node['rforecast']

        return
    
    
    """
    Helper function to compute the probability of a forecasted value
    """
    def _proba(self, node, values):
        
        index    = node['attribute']
        value    = node['value']
        operator = node['operator']

        if operator == '<':

            if values[index] < value:
                if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
                    prob_list = self._proba(node['left'], values)
                    return prob_list
                else:
                    indices = self.y_[node['lindices']]
            else:
                if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
                    prob_list = self._proba(node['right'], values)
                    return prob_list
                else:
                    indices = self.y_[node['rindices']]

        else:

            if values[index] != value:
                if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
                    prob_list = self._proba(node['left'], values)
                    return prob_list
                else:
                    indices = self.y_[node['lindices']]
            else:
                if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
                    prob_list = self._proba(node['right'], values)
                    return prob_list
                else:
                    indices = self.y_[node['rindices']]            

        prob_list = list()
        length = len(indices)
        
        for i in np.arange(self.n_classes):
            prob_list.append(np.sum(self.y_[indices] == i) / length)
            
        return prob_list


    """
    Compute the forecast values for a node
        
     * lindices - indices of the left dataset
     * rindices - indices of the right dataset
     
    Return the forecasted values
    """
    def _compute_forecast(self, lindices, rindices):
        
        lly = list(self.y_[lindices])
        lry = list(self.y_[rindices])
                  
        lforecast = max(set(lly), key=lly.count)
        rforecast = max(set(lry), key=lry.count)
                        
        return lforecast, rforecast


    """
    Helper function to recursively compute the variables in use
    """
    def _varinuse(self, node):
                        
        self.var_in_use_[node['attribute']] = 1
        
        # Process left branch
        if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
            self._varinuse(node['left'])
    
        # Process right branch    
        if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
            self._varinuse(node['right'])
                
        return

    
    """
    Compute the global miscoding of the dataset used by the current tree
    
    var_in_use: pre-computed attributes in use. If None, we compute the attributes using the current tree.

    Return the miscoding (float)
    """
    def _miscoding(self, attr=None):
                
        if attr is None:
            # No attribute provided, compute variables in use
            viu = self.var_in_use_
            self.var_in_use_ = np.zeros(self.X_.shape[1])
            self._varinuse(self.root_)
            var_in_use = self.var_in_use_
            self.var_in_use_ = viu

        else:
            # Use the pre-computed variables in use
            var_in_use = self.var_in_use_.copy()
            var_in_use[attr] = 1

        miscoding = np.dot(var_in_use, self.miscoding)
        miscoding = 1 - miscoding
                            
        return miscoding


    """
    Compute the inaccuracy of the current tree

    y_hat : pre-computed predictions. If None, we compute predictions using the current tree.

    Return the inaccuracy (float)
    """
    def _inaccuracy(self, y_hat=None):
        
        if y_hat is None:
            y_hat = self.y_hat_

        len_pred, len_joint = self._optimal_code_length(x1=y_hat, numeric1=False,
                                                        x2=self.y_, numeric2=False, double=True)
        inacc = ( len_joint - min(self.length_y, len_pred) ) / max(self.length_y, len_pred)

        return inacc 


    """
    Compute the surfeit of the current tree model
    
    Return the surfeit (float)
    """
    def _surfeit(self):
    
        # Compute the model string and its compressed version
        emodel = self._tree2str().encode()
        compressed = zlib.compress(emodel, level=1)
        
        km = len(compressed)
        lm = len(emodel)

        # Check if the model is too small to compress        
        if lm < km:
            return 1/2    # Experimental value for zlib

        if self.length_y < km:
            # surfeit = 1 - l(C(y)) / l(m)
            surfeit = 1 - self.length_y / lm
        else:
            # surfeit = 1 - l(m*) / l(m)
            surfeit = 1 - km / lm


        return surfeit


    """
    Compute the nescience of the current tree

    * y_hat:   Use pre-computed y_hat, so that computation is faster
    * attr:    Use pre-computed miscoding, so that computation is faster
    * surfeit: Use pre-computed surfeit, so that computation is faster
              
    Return the nescience (float)
    """
    def _nescience(self, y_hat=None, attr=None, surfeit=None):

        # Use pre-computed value or compute new ones

        if y_hat is None:
            inaccuracy = self._inaccuracy()
        else:
            inaccuracy = self._inaccuracy(y_hat)

        if attr is None:
            miscoding = self._miscoding()
        else:
            miscoding = self._miscoding(attr)

        surfeit = self._surfeit()

        # Avoid the inestability of inaccuracy            
        if surfeit < inaccuracy:
            surfeit = inaccuracy

        # Compute the nescience using an Euclidean distance
        nescience = math.sqrt(miscoding**2 + inaccuracy**2 + surfeit**2)

        return nescience


    """
    Split a dataset based on an attribute and an attribute value
    
      * attribute   - column number of the attribute
      * value       - value of the attribute for the split
      * indices     - array with the indices of the rows to split
      * numeric     - if the attribute is numeric or not
    
    Return
    
      * lindices - numpy array with those indices from the left side
      * rindices - numpy array with those indices from the rigth side
    """
    def _split_data(self, attribute, value, indices, numeric):

        if numeric:
            lindex = np.where(self.X_[:,attribute] < value)
        else:
            lindex = np.where(self.X_[:,attribute] == value)

        lindices = np.intersect1d(indices, lindex, assume_unique=True)
        rindices = np.setdiff1d(indices, lindices, assume_unique=True)

        return lindices, rindices


    """
    Create a new tree node based on the best split point
    Splitting criteria is inaccuracy
    
      * indices - indices of the rows to split
    
    Return a new node (dict)
    """   
    def _create_node(self, indices, constant_features):

        best_inaccuracy = 10e6    # A large value
        best_attribute  = None
        best_value      = None
        best_operator   = None
        best_lindices   = None
        best_rindices   = None
        best_lforecast  = None
        best_rforecast  = None

        my_constants = constant_features.copy()
        
        y = self.y_[indices]
        X = self.X_[indices]

        # Compute unique values and frequencies
        counts = dict()
        for i in range(len(y)):
            try:
                counts[y[i]] = counts[y[i]] + 1
            except KeyError:
                counts[y[i]] = 1

        # Do not split if all target points are equal
        if len(counts.keys()) <= 1:
            return CANNOT_BE_DEVELOPED

        # Search for the best split
        if self.n_jobs != 1:

            # In parallel
            valid_jobs = list()
            for i in np.arange(self.X_.shape[1]):
                if self.miscoding[i] < 0:
                    continue
                if constant_features[i]:
                   continue
                valid_jobs.append(i)
            results = Parallel(n_jobs=self.n_jobs)(delayed(self._create_node_helper)(i, self.X_isNumeric[i], X[:,i], y, counts) for i in valid_jobs)       

        else:

            # Sequential
            results = list()
            for i in np.arange(self.X_.shape[1]):
                if self.miscoding[i] < 0:
                    continue
                if constant_features[i]:
                    continue
                results.append(self._create_node_helper(i, self.X_isNumeric[i], X[:,i], y, counts))

        # Find out the best split
        for res in results:

            if res == CONSTANT_FEATURE:
                my_constants[i] == True
                continue

            if res == NO_SPLIT_FOUND:
                continue
            
            if res["inaccuracy"] < best_inaccuracy:
                best_inaccuracy = res["inaccuracy"]
                best_attribute  = res["attribute"]
                best_value      = res["value"]
                best_operator   = res["operator"]

        # Check if we have found a good split
        if best_attribute is None:
            return CANNOT_BE_DEVELOPED

        isnumeric = self.X_isNumeric[best_attribute]
        best_lindices, best_rindices = self._split_data(best_attribute, best_value, indices, isnumeric)
                    
        # Compute forecast
        lforecast, rforecast = self._compute_forecast(best_lindices, best_rindices)
        best_lforecast = lforecast
        best_rforecast = rforecast

        # Create the new node
        my_dict = {'attribute': best_attribute,
                   'value':     best_value,
                   'operator':  best_operator,
                   'left':      CAN_BE_DEVELOPED,
                   'right':     CAN_BE_DEVELOPED,
                   '_left':     CAN_BE_DEVELOPED,
                   '_right':    CAN_BE_DEVELOPED,                   
                   'lindices':  best_lindices,
                   'rindices':  best_rindices,
                   'lforecast': best_lforecast,
                   'rforecast': best_rforecast,
                   'constants': my_constants}
        
        return my_dict        


    """
    Helper function to create a new tree node
    """   
    def _create_node_helper(self, attribute, isNumeric, X, y, counts):

        best_inaccuracy = 10e6    # A large value
        best_i          = None
        best_value      = None
        best_operator   = None
                                                                
        # Create dictionaries to track the number of samples per class
        l_categories = defaultdict(int)
        r_categories = counts.copy()

        # Some datasets repeat the same values many times
        # values = set()

        # countX[val, cat] is the number of times that X[j] == values[i] and y[j] == c
        countX = defaultdict(dict) 
        for i in range(len(X)):
            # values.add(X[i])
            try:
                countX[X[i]][y[i]] = countX[X[i]][y[i]] + 1
            except KeyError:
                countX[X[i]][y[i]] = 1

        # values = list(values)
        values = list(dict.fromkeys(X))

        # We cannot use this attribute if it has no possible splitting points
        if len(values) <= 1:
            return CONSTANT_FEATURE

        # if isNumeric:
        #     values = sorted(values)

        # Search for the best splitting value
        l_tot = 0
        for i in np.arange(len(values)):

            val = values[i]

            # Update the categories
            if not isNumeric:
                l_categories = defaultdict(int)
                r_categories = counts.copy()

            for key in countX[val].keys():
                ct = countX[val][key]

                l_categories[key] = l_categories[key] + ct
                l_tot = l_tot + ct

                r_categories[key] = r_categories[key] - ct
            
            r_tot = len(X) - l_tot

            l_len = r_len = 0
            for tval in l_categories.values():
                if tval != 0 and l_tot != 0:
                    l_len = l_len + tval * ( - np.log2(tval / l_tot ))
            for tval in r_categories.values():
                if tval != 0 and r_tot != 0:
                    r_len = r_len + tval * ( - np.log2(tval / r_tot ))

            inaccuracy = l_len + r_len        

            # Check if we have found a better spliting point
            if inaccuracy < best_inaccuracy:
                best_inaccuracy = inaccuracy
                best_i          = i

        if not isNumeric:
            best_value    = values[best_i]
            best_operator = '='                        
        else:
            if best_i == (len(values)-1):
                # No good split has been found
                return NO_SPLIT_FOUND
            else:
                # Split using the middle point
                best_value    = (values[best_i] + values[best_i+1]) / 2
                best_operator = '<'                      

        # Create the new node
        my_dict = {'inaccuracy': best_inaccuracy,
                   'attribute':  attribute,
                   'value':      best_value,
                   'operator':   best_operator}
        
        return my_dict


    """
    Helper function to recursively compute the head of the tree
    """
    def _head2str(self, node, column_names=None):

        myset = set()

        # Print the attribute to take at this level
        if column_names is not None:
            myset.add(column_names[node['attribute']])            
        else:
            myset.add('X%d' % (node['attribute']))

        # Process left branch
        if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
            myset = myset.union(self._head2str(node['left'], column_names))
    
        # Process right branch    
        if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
            myset = myset.union(self._head2str(node['right'], column_names))
                
        return myset


    """
    Helper function to recursively compute the body of the tree
    """
    def _body2str(self, node, depth, max_depth=None, column_names=None):

        string = ""

        # Print the decision to take at this level
        if column_names is not None:
            if node['operator'] == '<':
                string = string + '%sif %s %s %.3f:\n' % (' '*depth*4, (column_names[node['attribute']]), node['operator'], node['value'])
            else:
                string = string + '%sif %s %s %s:\n' % (' '*depth*4, (column_names[node['attribute']]), node['operator'], node['value'])
        else:
            if node['operator'] == '<':
                string = string + '%sif X%d %s %.3f:\n' % (' '*depth*4, (node['attribute']), node['operator'], node['value'])
            else:
                string = string + '%sif X%d %s %s:\n' % (' '*depth*4, (node['attribute']), node['operator'], node['value'])

        # Process left branch
        if max_depth == depth:
            string = string + '%s[...]\n' % (' '*(depth+1)*4)
        else:
            if node['left'] == CAN_BE_DEVELOPED or node['left'] == CANNOT_BE_DEVELOPED:
                string = string + '%sreturn %s\n' % (' '*(depth+1)*4, self.classes_[node['lforecast']])
            else:
                string = string + self._body2str(node['left'], depth+1, max_depth, column_names)
    
        string = string + '%selse:\n' % (' '*depth*4)

        # Process right branch
        if max_depth == depth:
            string = string + '%s[...]\n' % (' '*(depth+1)*4)
        else:
            if node['right'] == CAN_BE_DEVELOPED or node['right'] == CANNOT_BE_DEVELOPED:
                string = string + '%sreturn %s\n' % (' '*(depth+1)*4, self.classes_[node['rforecast']])
            else:
                string = string + self._body2str(node['right'], depth+1, max_depth, column_names)
                
        return string

    
    """
    Convert a tree into a string
    
    Convert a decision tree into a string

      * depth - up to which level print the tree
      * column_names - use column names instead of "X" values

    In case of computing the surfeit of a tree, both values "depth" and
    "column_names" should be None.
        
    Return a string with a representation of the tree
    """
    def _tree2str(self, depth=None, column_names=None):
    
        # Compute the tree header
        string = "def tree" + str(self._head2str(self.root_, column_names)) + ":\n"

        # Compute the tree body
        string = string + self._body2str(self.root_, 1, depth, column_names)

        return string


    """
    Helper function to recursively compute the body of the tree in natural language
    """
    def _body2nl(self, node, depth, max_depth=None, column_names=None, categories=None):

        string = ""

        # Print the decision to take at this level
        if column_names is not None:
            if node['operator'] == '<':
                string = string + '%sif %s %s %.3f:\n' % (' '*depth*4, (column_names[node['attribute']]), node['operator'], node['value'])
            else:
                string = string + '%sif %s %s %s:\n' % (' '*depth*4, (column_names[node['attribute']]), node['operator'], node['value'])
        else:
            if node['operator'] == '<':
                string = string + '%sif X%d %s %.3f:\n' % (' '*depth*4, (node['attribute']), node['operator'], node['value'])
            else:
                string = string + '%sif X%d %s %s:\n' % (' '*depth*4, (node['attribute']), node['operator'], node['value'])

        # Process left branch
        if max_depth == depth or node['left'] == CAN_BE_DEVELOPED or node['left'] == CANNOT_BE_DEVELOPED:
            # Check if this branch contains one of the categories in which we are interested
            if self.classes_[node['lforecast']] in categories:
                string = string + '%sreturn %s ' % (' '*(depth+1)*4, self.classes_[node['lforecast']])
                lproba = np.sum(self.y_[node['lindices']] == node["lforecast"]) / len(node['lindices'])
                string = string + ' with probability %s\n' % (lproba)
            # else:
            #    string = string + '%s[...]\n' % (' '*(depth+1)*4)
        else:
            string = string + self._body2nl(node['left'], depth+1, max_depth, column_names, categories)
    
        string = string + '%selse:\n' % (' '*depth*4)

        # Process right branch
        if max_depth == depth or node['right'] == CAN_BE_DEVELOPED or node['right'] == CANNOT_BE_DEVELOPED:
            # Check if this branch contains one of the categories in which we are interested            
            if self.classes_[node['rforecast']] in categories:
                string = string + '%sreturn %s ' % (' '*(depth+1)*4, self.classes_[node['rforecast']])
                rproba = np.sum(self.y_[node['rindices']] == node["rforecast"]) / len(node['rindices'])
                string = string + ' with probability %s\n' % (rproba)
            # else:
            #    string = string + '%s[...]\n' % (' '*(depth+1)*4)
        else:
            string = string + self._body2nl(node['right'], depth+1, max_depth, column_names, categories)
                
        return string


    """
    Generate a string in natural language
    
    Generate a string in natural language with the derivations of the tree

      * categories   - categories in which we are interested
      * depth        - up to which level print the tree
      * column_names - use column names instead of "X" values
        
    Return a string with a natural language representation of the tree
    """
    def _nlg(self, depth=None, column_names=None, categories=None):
    
        return self._body2nl(self.root_, 1, depth, column_names, categories)


    """
    Helper function to recursively compute the number of nodes
    """
    def _bodycount(self, node):

        nodes = 1
        
        # Process left branch    
        if node['left'] == CAN_BE_DEVELOPED or node['left'] == CANNOT_BE_DEVELOPED:
            nodes = nodes + 1
        else:
            nodes = nodes + self._bodycount(node['left'])
    
        # Process right branch    
        if node['right'] == CAN_BE_DEVELOPED or node['right'] == CANNOT_BE_DEVELOPED:
            nodes = nodes + 1
        else:
            nodes = nodes + self._bodycount(node['right'])
                
        return nodes

    
    """
    Compute the total number of nodes of the tree
    
    Return total number of nodes (int)
    """
    def nodecount(self):
    
        return(self._bodycount(self.root_))
    
    
    """
    Helper function to recursively compute the depth of a tree
    """
    def _bodydepth(self, node):
        
        ldepth = 0
        rdepth = 0
        
        if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
            ldepth = self._bodydepth(node['left'])
        
        if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
            rdepth = self._bodydepth(node['right'])

        if ldepth > rdepth:
            return ldepth + 1
        else:
            return rdepth + 1


    """
    Compute the depth of the current tree (length of the longest branch)
    
    Return depth of the tree (int) 
    """
    def maxdepth(self):
    
        return(self._bodydepth(self.root_))

    """
    Clean up a tree removing redundant nodes, that is, nodes where both
    sides forecast the same value
    """
    def _cleanup(self, node, parent, side):

        # Signal if some nodes have been removed
        flag = False
        
        # Check if it is a terminal node
        if (node['left'] == CANNOT_BE_DEVELOPED) and (node['right'] == CANNOT_BE_DEVELOPED):
            # If both sides forecast the same value, clean it up
            if node['lforecast'] == node['rforecast']:
                if parent is not None:    # Avoid to clean up the root node
                    if side == 'left':
                        parent['left']     = None
                        parent['lindices'] = np.concatenate((node['lindices'], node['rindices']))
                        flag = True
                    else:
                        parent['right']    = None
                        parent['rindices'] = np.concatenate((node['lindices'], node['rindices']))
                        flag = True
                    
        else:
            if node['left'] != CAN_BE_DEVELOPED and node['left'] != CANNOT_BE_DEVELOPED:
                flag = self._cleanup(node['left'], node, "left")
        
            if node['right'] != CAN_BE_DEVELOPED and node['right'] != CANNOT_BE_DEVELOPED:
                flag2 = self._cleanup(node['right'], node, "right")
                flag  = flag or flag2

        return flag


class NescienceDecisionTreeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, cleanup=True, partial_miscoding=True, early_stop=True, n_jobs=1, verbose=False):
        """
        Initialization of the tree

          * cleanup(bool)     : If redundant leaf nodes are removed from final tree
          * n_jobs (int)      : Number of concurrent jobs
          * partial_miscoding : use partial miscoding instead of adjusted miscoding
          * early_stop        : stop the algorithm as soon as a good solution has been found
          * verbose (bool)    : If True, prints out additional information
        """

        self.tree = NescienceDecisionTree(verbose=verbose, mode="classification", cleanup=cleanup, partial_miscoding=partial_miscoding, early_stop=early_stop, n_jobs=n_jobs)


    def fit(self, X, y):
        """
        Fit a model (a tree) given a dataset
    
        X : array-like, shape (n_samples, n_attributes)
            Sample vectors from which to compute miscoding.
            array-like, numpy or pandas array in case of numerical attributes
            pandas array in case of mixed numeric and caregorical attributes
            
        y : array-like, shape (n_samples)
            The target values as numbers or strings.
       
        Return the fitted model
        """

        self.tree.fit(X, y)


    def predict(self, X):
        """
        Predict class given a dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        A list with the classes predicted
        """

        return(self.tree.predict(X))


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        An array of probabilities. The order of the list match the order
        the internal attribute classes_
        """

        return(self.tree.predict_proba(X))


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
        y : array-like, shape (n_samples)
    
        Returns
        -------
        One minus the mean error
        """

        return(self.tree.score(X, y))


class NescienceDecisionTreeRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, cleanup=True, partial_miscoding=True, early_stop=True, n_jobs=1, verbose=False):
        """
        Initialization of the tree

          * cleanup(bool)     : If redundant leaf nodes are removed from final tree
          * n_jobs (int)      : Number of concurrent jobs
          * partial_miscoding : use partial miscoding instead of adjusted miscoding
          * early_stop        : stop the algorithm as soon as a good solution has been found
          * verbose (bool)    : If True, prints out additional information
        """
        
        self.tree = NescienceDecisionTree(verbose=verbose, cleanup=cleanup, partial_miscoding=partial_miscoding, early_stop=early_stop, n_jobs=n_jobs, mode="regression")


    def fit(self, X, y):
        """
        Fit a model (a tree) given a dataset
    
        X : array-like, shape (n_samples, n_attributes)
            Sample vectors from which to compute miscoding.
            array-like, numpy or pandas array in case of numerical attributes
            pandas array in case of mixed numeric and caregorical attributes
            
        y : array-like, shape (n_samples)
            The target values as numbers or strings.
       
        Return the fitted model
        """

        self.tree.fit(X, y)


    def predict(self, X):
        """
        Predict class given a dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        A list with the classes predicted
        """
        
        return(self.tree.predict(X))


    def predict_proba(self, X):
        """
        Predict the probability of being in a class given a dataset
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
    
        Returns
        -------
        An array of probabilities. The order of the list match the order
        the internal attribute classes_
        """

        return(self.tree.predict_proba(X))


    def score(self, X, y):
        """
        Evaluate the performance of the current model given a test dataset

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
        y : array-like, shape (n_samples)
    
        Returns
        -------
        One minus the mean error
        """

        return(self.tree.score(X, y))
