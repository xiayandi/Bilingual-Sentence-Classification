__author__ = 'yandixia'

"""
This is the util module for some basic operations
"""


def insertDict(key, dictionary):
    """
    func: insert into dictionary. Add one to the key entry
    param: key: the inserted key
    param: dictionary: the inserted dictionary
    return: n/a
    """
    if key in dictionary:
        dictionary[key] += 1
    else:
        dictionary[key] = 1