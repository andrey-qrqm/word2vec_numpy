import numpy as np 
import re
from datetime import datetime

class Config:
    def __init__(
        self, 
        vocab_size, 
        emb_size, 
        learning_rate, 
        epochs, 
        batch_size, 
        print_cost
        ):
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.print_cost = print_cost

class Ids:
    def __init__(self, center_id, pos_id, neg_ids):
        self.center_id = center_id
        self.pos_id = pos_id
        self.neg_ids = neg_ids

class Gradients:
    def __init__(self, grad_v_c, grad_v_p, grad_V_n):
        self.grad_v_c = grad_v_c
        self.grad_v_p = grad_v_p
        self.grad_V_n = grad_V_n

def get_text(filename):
    with open(filename) as f:
        data = f.read()
    return data


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

def initialize_wrd_emb(vocab_size, emb_size):
    """
    vocab_size: int. vocabulary size of your corpus or training data
    emb_size: int. word embedding size. How many dimensions to represent each vocabulary
    """
    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
    return WRD_EMB

def initialize_dense(input_size, output_size):
    """
    input_size: int. size of the input to the dense layer
    output_szie: int. size of the output out of the dense layer
    """
    W = np.random.randn(output_size, input_size) * 0.01
    return W

def initialize_parameters(vocab_size, emb_size):
    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
    W = initialize_dense(emb_size, vocab_size)
    parameters = {}
    parameters['WRD_EMB'] = WRD_EMB
    parameters['W'] = W
    
    return parameters

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def negative_sampling_loss(Id, parameters):
    """
    Id.center_id : int       — center word index
    Id.pos_id    : int       — positive neighbor index
    Id.neg_ids   : (K,)      — indices of K negative words
    """
    WRD_EMB = parameters['WRD_EMB']  # (vocab_size, emb_size)

    v_c = WRD_EMB[Id.center_id]   # (emb_size,) — center word vector 
    v_p = WRD_EMB[Id.pos_id]      # (emb_size,) — positive neighbor vector
    V_n = WRD_EMB[Id.neg_ids]     # (K, emb_size) — negative neighbor vectors
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
    Gradient = Gradients(grad_v_c, grad_v_p, grad_V_n)

    return loss, Gradient

def update_parameters_ns(parameters, Ids, Gradient, learning_rate):
    parameters['WRD_EMB'][Ids.center_id] -= learning_rate * Gradient.grad_v_c
    parameters['WRD_EMB'][Ids.pos_id]    -= learning_rate * Gradient.grad_v_p
    parameters['WRD_EMB'][Ids.neg_ids]   -= learning_rate * Gradient.grad_V_n

def skipgram_model_training_ns(X, Y, negative_samples, config, parameters):
    costs = []
    m = X.shape[1]

    if parameters is None:
        parameters = initialize_parameters(config.vocab_size, config.emb_size)

    begin_time = datetime.now()
    for epoch in range(config.epochs):
        epoch_cost = 0
        # Shuffle
        perm = np.random.permutation(m)

        for batch_start in range(0, m, config.batch_size):
            batch_idx = perm[batch_start:batch_start + config.batch_size]

            batch_cost = 0
            for idx in batch_idx:
                center_id = X[0, idx]
                pos_id = Y[0, idx]
                neg_ids = negative_samples[idx]
                Id = Ids(center_id, pos_id, neg_ids)
                loss, Gradient = negative_sampling_loss(Id, parameters)
                update_parameters_ns(
                    parameters, 
                    Id, 
                    Gradient, 
                    config.learning_rate
                )
                batch_cost += loss
            epoch_cost += batch_cost / len(batch_idx)
        costs.append(epoch_cost)

        if config.print_cost and epoch % max(1, config.epochs // 10) == 0:
            print("Cost after epoch {}: {:.4f}".format(epoch, epoch_cost))
        if epoch % max(1, config.epochs // 100) == 0:
            config.learning_rate *= 0.98
    end_time = datetime.now()
    print('Training time: {}'.format(end_time - begin_time))
    return parameters

if __name__ == "__main__":
    text = get_text('input.txt')
    tokens = tokenize(text)
    word_to_id, id_to_word = mapping(tokens)
    vocab_size = len(word_to_id)
    X, Y, negative_samples = generate_training_data(tokens, word_to_id, window_size=2, negative_samples_size=5)
    config = Config(        
        vocab_size=vocab_size,
        emb_size=50,
        learning_rate=0.05,
        epochs=1000,
        batch_size=128,
        print_cost=True
    )

    params = skipgram_model_training_ns(
        X, Y, negative_samples, config,
        parameters=None,
    )
