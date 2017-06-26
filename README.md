# Fitting-Autoencoders-to-DNA
Model dimensions are not working out: Received error message "lstm_1 to have 2 dimensions, but got array with shape (439763, 10, 4)". LSTM_1 is the first layer of the model...

Vectorization works out, the training data has shape (439763, 10, 4), which is in the format (n_samples, len_of_str, num_of_chars). This is the same format as the training data for the addition_rnn.py. 


