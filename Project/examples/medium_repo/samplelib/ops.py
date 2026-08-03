def normalize(values):
    total = sum(values) or 1
    return [value / total for value in values]


def clamp(value, low=0, high=1):
    if value < low:
        return low
    if value > high:
        return high
    return value
