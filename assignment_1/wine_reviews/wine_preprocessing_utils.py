import pandas as pd
import numpy as np
import pycountry


def country_to_code(name: str) -> str:
    """
    Get country code from name.
    Replace '?' with None.
    """
    
    try:
        return pycountry.countries.lookup(name).alpha_2
    except LookupError:
        if name == "England":
            return "GB"
        elif name == "Turkey":
            return "TR"
        elif name == "Macedonia":
            return "MK"
        else:
            return "UNK"