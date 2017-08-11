Files include: 2 Layer LSTM model (based on the char_rnn keras example) to recreate DNA base strings; 1 and 2 Layer LSTM models to predict the next amino acid (based on the Nietzsche text generator keras example); Embedding layer + 1 layer LSTM model to pedict the next amino acid; DNA Variational Autoencoder (based on the variational autoencoder keras example) 

AA_CVAE: still trying to encode the labels (function_index) 
AA_VAE: incomplete
ALSTM: epoch computation time >225 sec
Basic_DNA_Autoencoder: ignore
DNA_VAE: main model for the VAE. encoder = 2 hidden layers, decoder = 1 hidden layer
DNA_VAE_Custom Accuracy: not working
DNA_VAE_LRSchedule: implemented a lr scheduling/lr plateau callback
DNA_VAE_withLSTM: has lower acc/corr than the regular VAE model
Protein_Gen_LSTM: 2 layer LSTM model 
Protein_Gen_LSTM_Embed.py: Embedding layer + 2 layer LSTM model 
