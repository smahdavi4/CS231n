from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        
        self.params = {
            'W1': np.random.normal(0.0, weight_scale, (input_dim, hidden_dim)),
            'b1': np.zeros((hidden_dim, )),
            'W2':np.random.normal(0.0, weight_scale, (hidden_dim, num_classes)),
            'b2': np.zeros((num_classes, ))
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out_hidden, cache_hidden = affine_forward(X, self.params['W1'], self.params['b1'])
        out_relu, cache_relu = relu_forward(out_hidden)
        out_class, cache_class = affine_forward(out_relu, self.params['W2'], self.params['b2'])
        scores = out_class

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, dx = softmax_loss(scores, y)
        dx_hidden, dw_hidden, db_hidden = affine_backward(dx, cache_class)
        dx_relu = relu_backward(dx_hidden, cache_relu)
        dx_input, dw_input, db_input = affine_backward(dx_relu, cache_hidden)
        
        loss += self.reg * 0.5 * (np.sum(self.params['W1'] * self.params['W1']) + np.sum(self.params['W2'] * self.params['W2']))
        
        grads = {
            'W1': dw_input + self.reg * self.params['W1'],
            'b1': db_input,
            'W2': dw_hidden + self.reg * self.params['W2'],
            'b2': db_hidden
        }

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        layers = [input_dim] + hidden_dims + [num_classes]
        
        for i in range(1, len(layers)):
            self.params.update({
                'b{}'.format(i): np.zeros((layers[i], )),
                'W{}'.format(i): np.random.normal(0.0, weight_scale, (layers[i-1], layers[i])),
            })
        if self.normalization in ["batchnorm", "layernorm"]:
            for i in range(1, len(layers) - 1):
                self.params.update({
                    'gamma{}'.format(i): np.ones((layers[i], )),
                    'beta{}'.format(i): np.zeros((layers[i], )),
                })
                
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        current_X = X
        out_hiddens = [None] * (self.num_layers)
        cache_hiddens = [None] * (self.num_layers)
        out_relus = [None] * (self.num_layers)
        cache_relus = [None] * (self.num_layers)
        
        out_dropouts = [None] * (self.num_layers)
        cache_dropouts = [None] * (self.num_layers)
        
        out_batchnorms = [None] * (self.num_layers)
        cache_batchnorms = [None] * (self.num_layers)
        
        for i in range(1, self.num_layers):
            out_hiddens[i], cache_hiddens[i] = affine_forward(current_X, self.params['W{}'.format(i)], self.params['b{}'.format(i)])
            
            # Batch/Layer Norm
            out_hid = out_hiddens[i]
            if self.normalization in ["batchnorm", "layernorm"]:
                norm_forward = batchnorm_forward if self.normalization == "batchnorm" else layernorm_forward
                gamma, beta = self.params['gamma{}'.format(i)], self.params['beta{}'.format(i)]
                out_batchnorms[i], cache_batchnorms[i] = norm_forward(out_hid, gamma, beta, self.bn_params[i-1])
                out_hid = out_batchnorms[i]

            # Relu
            out_relus[i], cache_relus[i] = relu_forward(out_hid)
            
            # Dropout
            if self.use_dropout:
                out_dropouts[i], cache_dropouts[i] = dropout_forward(out_relus[i], self.dropout_param)
                current_X = out_dropouts[i]
            else:
                current_X = out_relus[i]
            
        out_class, cache_class = affine_forward(current_X, self.params['W{}'.format(self.num_layers)], self.params['b{}'.format(self.num_layers)])
        scores = out_class

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        dx_hiddens, dw_hiddens, db_hiddens = [None]*self.num_layers, [None]*self.num_layers, [None]*self.num_layers
        dgammas, dbetas = [None]*self.num_layers, [None]*self.num_layers
        
        loss, dx = softmax_loss(scores, y)
        current_dx = dx
        dx_hiddens[-1], dw_hiddens[-1], db_hiddens[-1] = affine_backward(dx, cache_class)
        
        for i in range(self.num_layers-2, -1, -1):
            # Dropout & Relu
            if self.use_dropout:
                dx_dropout = dropout_backward(dx_hiddens[i+1], cache_dropouts[i+1])
                dx_relu = relu_backward(dx_dropout, cache_relus[i+1])
            else:
                dx_relu = relu_backward(dx_hiddens[i+1], cache_relus[i+1])
            
            # Batchnorm
            if self.normalization in ["batchnorm", "layernorm"]:
                norm_backward = batchnorm_backward if self.normalization == "batchnorm" else layernorm_backward
                dx_batchnorm, dgamma, dbeta = norm_backward(dx_relu, cache_batchnorms[i+1])
                dgammas[i] = dgamma
                dbetas[i] = dbeta
                dx_hiddens[i], dw_hiddens[i], db_hiddens[i] = affine_backward(dx_batchnorm, cache_hiddens[i+1])
            else:
                dx_hiddens[i], dw_hiddens[i], db_hiddens[i] = affine_backward(dx_relu, cache_hiddens[i+1])
            
            
        loss += self.reg * 0.5 * (np.sum([np.sum(self.params['W{}'.format(i)] * self.params['W{}'.format(i)]) for i in range(1, self.num_layers)]))
        
        for i in range(self.num_layers):
            grads.update({
                'W{}'.format(i+1): dw_hiddens[i] + self.reg * self.params['W{}'.format(i+1)],
                'b{}'.format(i+1): db_hiddens[i],
            })
        
        if self.normalization in ["batchnorm", "layernorm"]:
            for i in range(self.num_layers - 1):
                grads.update({
                    'gamma{}'.format(i+1): dgammas[i],
                    'beta{}'.format(i+1): dbetas[i]
                })

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
