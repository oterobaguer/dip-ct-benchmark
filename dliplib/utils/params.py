import os
import json

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Params:
    """
    Class that stores and loads hyper-parameters
    """
    def __init__(self, params=None):
        if params:
            self.__dict__.update(params)

    def save(self, json_key):
        """
        Saves parameters to json file
        :param json_key: Key that identifies the params
        """
        json_path = os.path.join(BASE_DIR, 'utils/params', json_key + '.json')
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @staticmethod
    def load(json_key):
        """
        Loads parameters from json file
        :param json_key: Key that identifies the params
        """
        json_path = os.path.join(BASE_DIR, 'utils/params', json_key + '.json')
        with open(json_path) as f:
            params = json.load(f)
            return Params(params)

    @property
    def dict(self):
        """
        Returns a dictionary structure with the values of the parameters
        """
        return self.__dict__