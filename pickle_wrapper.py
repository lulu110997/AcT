import pickle

def open_pickle(pickle_path):
    """
    Utility function for deserialising a pickle file
    Args:
        pickle_path: String | Path to pickle file

    Returns:
        pickle_file: pickle file that was deserialised
    """
    with open(pickle_path, 'rb') as f:
        pickle_file = pickle.load(f)

    return pickle_file


def save_pickle(pickle_name, data):
    """
    Save the data as a pickle file

    Args:
        pickle_name: String | Filename including the path
        data: data to be saved as a pickle

    Returns:
    """
    if not pickle_name.endswith('.pickle'):
        pickle_name = pickle_name + '.pickle'

    with open(pickle_name, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
