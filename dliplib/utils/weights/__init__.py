import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_weights_path(weights_key):
    path = os.path.join(BASE_DIR, 'weights', weights_key + '.pt')
    return path


def save_weights(reconstructor, weights_key, **kwargs):
    """
    Saves parameters to a file
    :param model: PyTorch model that will save the weights
    :param weights_key: Key that identifies the weights
    """
    path = get_weights_path(weights_key)
    reconstructor.save_learned_params(path, **kwargs)


def load_weights(reconstructor, weights_key, **kwargs):
    """
    Loads weights from file
    :param model: PyTorch model that will load the weights
    :param weights_key: Key that identifies the weights
    """
    path = get_weights_path(weights_key)
    reconstructor.load_learned_params(path, **kwargs)
