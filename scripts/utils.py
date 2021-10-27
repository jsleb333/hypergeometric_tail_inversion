import os
import __main__

def get_path(file):
    return os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')


main_path = get_path(__main__.__file__)
