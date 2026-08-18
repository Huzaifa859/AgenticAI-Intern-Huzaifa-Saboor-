from .ops import clamp, normalize


class Scorer:
    def score(self, values):
        weights = normalize(values)
        return clamp(sum(weights))


def broken_call():
    return clamp(0.5, 0, 1, 2)
