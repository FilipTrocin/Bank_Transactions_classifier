import re


def remove_numbers(array):
    modified = []
    for x in array:
        wo_date = re.sub('\d\w.*\d', "", x.lower())
        wo_pymt_type = re.sub('\s\d.*', "", wo_date)
        modified.append(wo_pymt_type)
    return modified


def remove_pymt_types(array):
    modified = []
    for x in array:
        wo_pymt_types = x.replace("on", "").replace("contactless", "").replace("sundry debit", "") \
            .replace("card payment", "").replace("ctactless", "").replace("direct debit", "") \
            .replace("cash at transact", "").replace("via faster payment", "").replace("via mobile", "") \
            .replace("apple pay", "").replace("faster payments out", "").replace("line", "")
        modified.append(wo_pymt_types)
    return modified


def remove_extra_symbols(array):
    modified = []
    for x in array:
        wo_extra_space = x.replace("   ", " ").replace("  ", " ")
        wo_colon = re.sub('.*:', "", wo_extra_space).strip()
        modified.append(wo_colon)
    return modified




