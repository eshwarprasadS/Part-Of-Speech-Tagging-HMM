# Part-Of-Speech-Tagging-HMM
Using Hidden Markov Model for POS Tagging in any unknown language

Usage : 

- Run the Learning Python Script with the training data's directory path as command line argument
- The learning script will create a params file called 'hmmmodel.txt' which is a human readable text file containing the HMM parameters
- Run the Decoding Script with the test data directory path as CL argument. 
- The Decoding script reads the hmm model from the saved file and applies VITERBI DECODING to assign POS tags to the test data.
