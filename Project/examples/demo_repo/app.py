"""Demo module with several intentional static-analysis defects."""

import os
import json


def duplicate():
    """First definition — discarded by the second."""
    return 1


def duplicate():
    """Duplicate definition in the same scope."""
    return 2


def collect(name, bucket=[]):
    """Append to a shared default list."""
    bucket.append(name)
    return bucket


def risky():
    """Call an undefined name inside a bare except."""
    try:
        return undefined_helper()
    except:
        return None
