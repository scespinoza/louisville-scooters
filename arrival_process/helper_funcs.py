import pickle

def save_pickle(temporal_dict, filename):
    output = open(filename, 'wb')
    pickle.dump(temporal_dict, output, protocol=pickle.HIGHEST_PROTOCOL)
    output.close()
    
def load_pickle(filename):
     pkl_file = open(filename, 'rb')
     myfile = pickle.load(pkl_file)
     pkl_file.close()
     return myfile

def load_pickle_from_py2(filename, encoding='latin1'):
     pkl_file = open(filename, 'rb')
     myfile = pickle.load(pkl_file, encoding=encoding)
     pkl_file.close()
     return myfile