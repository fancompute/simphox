from .typing import Excitation, SourceLabel


def parse_excitation(excitation: Excitation):
    """Parse any excitation format into a list of tuples consisting of port name and mode index.

    Args:
        excitation: Excitation of various types

    Returns:
        List of tuples consisting of port name and mode index.

    """
    if isinstance(excitation, str):
        return [(excitation, 0)]
    elif isinstance(excitation, tuple) or isinstance(excitation, list):
        if (isinstance(excitation[0], str) or isinstance(excitation[0], int)) and isinstance(excitation[1], int) and len(excitation) == 2:
            return [excitation]
        return sum([parse_excitation(mi) for mi in excitation], [])
    elif isinstance(excitation, dict):
        return sum([parse_excitation([(mi, idx) for idx in excitation[mi]]) for mi in excitation], [])


def parse_source_port(source: SourceLabel):
    """Parse any acceptable source format into a dict between tuples of port name and mode index and weight.

    Args:
        source: Source of various types/formats

    Returns:
        Dictionary of tuples consisting of port name and mode index mapped to weights.

    """
    if isinstance(source, str):
        return {(source, 0): 1}
    elif isinstance(source, tuple) or isinstance(source, list):
        if isinstance(source[0], str) and isinstance(source[1], int) and len(source) == 2:
            return {tuple(source): 1}
        return {k: v for mi in source for k, v in parse_source_port(mi).items()}
    elif isinstance(source, dict):
        return {(s, 0) if isinstance(s, str) else s: w for s, w in source.items()}
