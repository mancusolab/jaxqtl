import sys


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)
