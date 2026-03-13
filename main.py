import numpy as np 
import re


def get_text(filename):
    with open(filename) as f:
        data = f.read()
    return data

print(get_text('input.txt'))

#Implementation for skip_gram model
def tokenize(text):
    # obtains tokens with a least 1 alphabet
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())

def mapping(tokens):
    word_to_id = dict()
    id_to_word = dict()

    for i, token in enumerate(set(tokens)):
        word_to_id[token] = i
        id_to_word[i] = token

    return word_to_id, id_to_word


def generate_training_data(tokens, word_to_id, window_size, negative_samples_size):
    N = len(tokens)
    X, Y, negative_samples = [], [], []

    all_ids = set(word_to_id.values())

    for i in range(N):
        nbr_inds = list(range(max(0, i - window_size), i)) + \
                   list(range(i + 1, min(N, i + window_size + 1)))

        excluded_ids = set(word_to_id[tokens[k]] for k in nbr_inds)
        excluded_ids.add(word_to_id[tokens[i]])

        available_ids = list(all_ids - excluded_ids)

        for j in nbr_inds:
            X.append(word_to_id[tokens[i]])
            Y.append(word_to_id[tokens[j]])
            # for each pair we generate negative samples 
            neg = np.random.choice(available_ids, negative_samples_size, replace=False)
            negative_samples.append(neg)

    X = np.array(X) # Shape: (1, m)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y) # Shape: (1, m)
    Y = np.expand_dims(Y, axis=0)
    negative_samples = np.array(negative_samples) # Shape: (m, negative_samples_size)
    print("x shape: {}, y shape: {}, negative samples shape: {}".format(X.shape, Y.shape, negative_samples.shape))
    return X, Y, negative_samples

text = get_text('input.txt')
tokens = tokenize(text)
word_to_id, id_to_word = mapping(tokens)
vocab_size = len(word_to_id)
X, Y, negative_samples = generate_training_data(tokens, word_to_id, window_size=2, negative_samples_size=5)

m = Y.shape[1]
# turn Y into one hot encoding
Y_one_hot = np.zeros((vocab_size, m))
Y_one_hot[Y.flatten(), np.arange(m)] = 1




def initialize_wrd_emb(vocab_size, emb_size):
    """
    vocab_size: int. vocabulary size of your corpus or training data
    emb_size: int. word embedding size. How many dimensions to represent each vocabulary
    """
    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    
    assert(WRD_EMB.shape == (vocab_size, emb_size))
    return WRD_EMB

def initialize_dense(input_size, output_size):
    """
    input_size: int. size of the input to the dense layer
    output_szie: int. size of the output out of the dense layer
    """
    W = np.random.randn(output_size, input_size) * 0.01
    
    assert(W.shape == (output_size, input_size))
    return W

def initialize_parameters(vocab_size, emb_size):
    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)
    
    parameters = {}
    parameters['WRD_EMB'] = WRD_EMB
    parameters['W'] = W
    
    return parameters

def ind_to_word_vecs(inds, parameters):
    """
    inds: numpy array. shape: (1, m)
    parameters: dict. weights to be trained
    """
    m = inds.shape[1]
    WRD_EMB = parameters['WRD_EMB']
    word_vec = WRD_EMB[inds.flatten(), :].T
    
    assert(word_vec.shape == (WRD_EMB.shape[1], m))
    
    return word_vec

def linear_dense(word_vec, parameters):
    """
    word_vec: numpy array. shape: (emb_size, m)
    parameters: dict. weights to be trained
    """
    m = word_vec.shape[1]
    W = parameters['W']
    Z = np.dot(W, word_vec)
    
    assert(Z.shape == (W.shape[0], m))
    
    return W, Z

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def negative_sampling_loss(center_id, pos_id, neg_ids, parameters):
    """
    center_id : int       — center word index
    pos_id    : int       — positive neighbor index
    neg_ids   : (K,)      — indices of K negative words
    """
    WRD_EMB = parameters['WRD_EMB']  # (vocab_size, emb_size)

    v_c = WRD_EMB[center_id]   # (emb_size,) — center word vector 
    v_p = WRD_EMB[pos_id]      # (emb_size,) — positive neighbor vector
    V_n = WRD_EMB[neg_ids]     # (K, emb_size) — negative neighbor vectors

    score_pos = np.dot(v_c, v_p)          # scalar
    score_neg = V_n @ v_c                 # (K,)

    loss = -np.log(sigmoid(score_pos) + 1e-7) \
           -np.sum(np.log(sigmoid(-score_neg) + 1e-7))

    # Gradients
    d_pos  = sigmoid(score_pos) - 1                    # Scalar
    d_neg  = sigmoid(score_neg)                        # (K,)

    grad_v_c = d_pos * v_p + d_neg @ V_n              # (emb_size,)
    grad_v_p = d_pos * v_c                             # (emb_size,)
    grad_V_n = np.outer(d_neg, v_c)                   # (K, emb_size)

    return loss, grad_v_c, grad_v_p, grad_V_n, neg_ids


def softmax(Z):
    """
    Z: output out of the dense layer. shape: (vocab_size, m)
    """
    softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)
    
    assert(softmax_out.shape == Z.shape)

    return softmax_out

def forward_propagation(inds, parameters):
    word_vec = ind_to_word_vecs(inds, parameters)
    W, Z = linear_dense(word_vec, parameters)
    softmax_out = softmax(Z)
    
    caches = {}
    caches['inds'] = inds
    caches['word_vec'] = word_vec
    caches['W'] = W
    caches['Z'] = Z
    
    return softmax_out, caches

def cross_entropy(softmax_out, Y):
    """
    softmax_out: output out of softmax. shape: (vocab_size, m)
    """
    m = softmax_out.shape[1]
    cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
    return cost

def softmax_backward(Y, softmax_out):
    """
    Y: labels of training data. shape: (vocab_size, m)
    softmax_out: output out of softmax. shape: (vocab_size, m)
    """
    dL_dZ = softmax_out - Y
    
    assert(dL_dZ.shape == softmax_out.shape)
    return dL_dZ

def dense_backward(dL_dZ, caches):
    """
    dL_dZ: shape: (vocab_size, m)
    caches: dict. results from each steps of forward propagation
    """
    W = caches['W']
    word_vec = caches['word_vec']
    m = word_vec.shape[1]
    
    dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
    dL_dword_vec = np.dot(W.T, dL_dZ)

    assert(W.shape == dL_dW.shape)
    assert(word_vec.shape == dL_dword_vec.shape)
    
    return dL_dW, dL_dword_vec

def backward_propagation(Y, softmax_out, caches):
    dL_dZ = softmax_backward(Y, softmax_out)
    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)
    
    gradients = dict()
    gradients['dL_dZ'] = dL_dZ
    gradients['dL_dW'] = dL_dW
    gradients['dL_dword_vec'] = dL_dword_vec
    
    return gradients

def update_parameters(parameters, caches, gradients, learning_rate):
    vocab_size, emb_size = parameters['WRD_EMB'].shape
    inds = caches['inds']
    dL_dword_vec = gradients['dL_dword_vec']
    m = inds.shape[-1]
    
    parameters['WRD_EMB'][inds.flatten(), :] -= dL_dword_vec.T * learning_rate

    parameters['W'] -= learning_rate * gradients['dL_dW']
    
def update_parameters_ns(parameters, center_id, pos_id,
                          grad_v_c, grad_v_p, grad_V_n,
                          neg_ids, learning_rate):
    parameters['WRD_EMB'][center_id] -= learning_rate * grad_v_c
    parameters['WRD_EMB'][pos_id]    -= learning_rate * grad_v_p
    parameters['WRD_EMB'][neg_ids]   -= learning_rate * grad_V_n

from datetime import datetime

import matplotlib.pyplot as plt


def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size=256, parameters=None, print_cost=False, plot_cost=True):
    costs = []
    m = X.shape[1]
    
    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)
    
    begin_time = datetime.now()
    for epoch in range(epochs):
        epoch_cost = 0
        batch_inds = list(range(0, m, batch_size))
        np.random.shuffle(batch_inds)
        for i in batch_inds:
            X_batch = X[:, i:i+batch_size]
            Y_batch = Y[:, i:i+batch_size]

            softmax_out, caches = forward_propagation(X_batch, parameters)
            gradients = backward_propagation(Y_batch, softmax_out, caches)
            update_parameters(parameters, caches, gradients, learning_rate)
            cost = cross_entropy(softmax_out, Y_batch)
            epoch_cost += np.squeeze(cost)
            
        costs.append(epoch_cost)
        if print_cost and epoch % (epochs // 500) == 0:
            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
        if epoch % (epochs // 100) == 0:
            learning_rate *= 0.98
    end_time = datetime.now()
    print('training time: {}'.format(end_time - begin_time))
            
    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
        plt.show()
    return parameters

def skipgram_model_training_ns(X, Y, negative_samples, vocab_size, emb_size,
                                learning_rate, epochs, batch_size=256,
                                parameters=None, print_cost=False, plot_cost=True):
    costs = []
    m = X.shape[1]

    if parameters is None:
        parameters = initialize_parameters(vocab_size, emb_size)

    begin_time = datetime.now()
    for epoch in range(epochs):
        epoch_cost = 0

        # Shuffle
        perm = np.random.permutation(m)

        for batch_start in range(0, m, batch_size):
            batch_idx = perm[batch_start:batch_start + batch_size]

            batch_cost = 0
            for idx in batch_idx:
                center_id = X[0, idx]
                pos_id    = Y[0, idx]
                neg_ids   = negative_samples[idx]

                loss, grad_v_c, grad_v_p, grad_V_n, neg_ids = negative_sampling_loss(
                    center_id, pos_id, neg_ids, parameters
                )
                update_parameters_ns(
                    parameters, center_id, pos_id,
                    grad_v_c, grad_v_p, grad_V_n,
                    neg_ids, learning_rate
                )
                batch_cost += loss

            epoch_cost += batch_cost / len(batch_idx)

        costs.append(epoch_cost)

        if print_cost and epoch % max(1, epochs // 500) == 0:
            print("Cost after epoch {}: {:.4f}".format(epoch, epoch_cost))

        if epoch % max(1, epochs // 100) == 0:
            learning_rate *= 0.98

    end_time = datetime.now()
    print('Training time: {}'.format(end_time - begin_time))

    if plot_cost:
        plt.plot(np.arange(epochs), costs)
        plt.xlabel('# of epochs')
        plt.ylabel('cost')
        plt.show()

    return parameters
"""
paras = skipgram_model_training_ns(X, Y_one_hot, negative_samples, vocab_size, 50, 0.05, 1000, batch_size=128, parameters=None, print_cost=True)

X_test = np.arange(vocab_size)
X_test = np.expand_dims(X_test, axis=0)
softmax_test, _ = forward_propagation(X_test, paras)
top_sorted_inds = np.argsort(softmax_test, axis=0)[-4:,:]

for input_ind in range(vocab_size):
    input_word = id_to_word[input_ind]
    output_words = [id_to_word[output_ind] for output_ind in top_sorted_inds[::-1, input_ind]]
    print("{}'s neighbor words: {}".format(input_word, output_words))
"""

X, Y, negative_samples = generate_training_data(tokens, word_to_id, window_size=2, negative_samples_size=5)

params = skipgram_model_training_ns(
    X, Y, negative_samples,
    vocab_size=vocab_size,
    emb_size=50,
    learning_rate=0.05,
    epochs=1000,
    batch_size=128,
    print_cost=True
)
