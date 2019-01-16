import collections


def to_sorted_dict(_dict):
    ret_dict = collections.OrderedDict()
    for k, v in sorted(_dict.items()):
        ret_dict[k] = sorted(v)

    return ret_dict
