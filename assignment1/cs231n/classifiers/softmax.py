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
    dW = np.zeros(W.shape) # dW = D*C

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_classes = W.shape[1] # C
    num_train = X.shape[0] # N
    
    for i in range(num_train):
        scores = X[i].dot(W) # (1,D)*(D,C) = (1,C)
        scores -= np.max(scores) # numeric instability
        
        scores = np.exp(scores) # exp 취함
        normalize = scores/np.sum(scores) # exp 취한 값 다 더한 걸로 각각 나누기
        
        loss_i = -np.log(normalize[y[i]])
        loss += loss_i
        
        for j in range(num_classes):           
            dW[:, j] += X[i] * normalize[j] # sj 더해줌
        dW[:, y[i]] -= X[i] # syi(정답) input image의 pixel만큼 빼줌
        
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

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
    
    num_train = X.shape[0] # N
    scores = X.dot(W) # (N,D)*(D,C) = (N,C)
    scores -= np.max(scores, axis=1, keepdims=True) # numeric instability

    scores = np.exp(scores) # exp 취함
    scores_sum = scores.sum(axis=1, keepdims=True) # 다 더한 거
    normalize = scores/scores_sum # 다 더한 걸로 나눠주기
    correct = normalize[np.arange(num_train), y]

    loss = np.sum(-np.log(correct))

    # Grdient
    normalize[np.arange(num_train), y] -= 1 # 정답이면 해당 row번째 사진의 pixel value만큼 빼주기
    dW = X.T.dot(normalize)

    loss /= num_train
    dW /= num_train
    
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
