from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_samples):
        true_class = y[i]
        y_pred = X[i].dot(W)
        y_pred_softmax = np.exp(y_pred) / np.sum(np.exp(y_pred))
        loss += -np.log(y_pred_softmax[true_class])
        
        a = y_pred
        da_dW = np.exp(a) / np.sum(np.exp(a))
        da_dW[true_class] -= 1.0
        dW += X[i].reshape((-1, 1)).dot(da_dW.reshape((1, -1)))

    dW /= num_samples
    loss /= num_samples
    loss += np.sum(reg * W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    num_classes = W.shape[1]
    
    true_classes = y
    y_preds = X.dot(W)
    y_preds_softmax = np.exp(y_preds) / np.sum(np.exp(y_preds), axis=1).reshape((-1, 1))
    loss += -np.sum(np.log(y_preds_softmax[np.arange(num_samples), true_classes]))
    
    a = y_preds
    da_dW = np.exp(a) / np.sum(np.exp(a), axis=1).reshape((-1 ,1))
    da_dW[np.arange(num_samples), true_classes] -= 1.0
    dW += X.transpose().dot(da_dW)

    dW /= num_samples
    loss /= num_samples
    loss += np.sum(reg * W * W)
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
