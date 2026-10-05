"""Fixed two-class rank blend; CSV tokens of the other ten findings are retained."""
import csv
import io
import math

TARGETS = ('Medial Meniscus', 'Lateral Meniscus')


def ranks(values):
    ordered = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    i = 0
    while i < len(ordered):
        j = i + 1
        while j < len(ordered) and values[ordered[j]] == values[ordered[i]]:
            j += 1
        for index in ordered[i:j]:
            out[index] = (i + 1 + j) / (2 * len(values))
        i = j
    return out


def parse(text, expected_columns=None):
    rows = list(csv.reader(io.StringIO(text)))
    if not rows or len(rows) < 2:
        raise ValueError('Empty predictions')
    header, rows = rows[0], rows[1:]
    if header[0] != 'StudyInstanceUID' or len(set(header)) != len(header):
        raise ValueError('Invalid header')
    if expected_columns and header != expected_columns:
        raise ValueError('Column order mismatch')
    if any(len(row) != len(header) or not row[0] for row in rows):
        raise ValueError('Invalid row')
    if len({row[0] for row in rows}) != len(rows):
        raise ValueError('Duplicate study')
    for row in rows:
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in row[1:]):
            raise ValueError('Nonfinite or out-of-range prediction')
    return header, rows


def overlay(parent_text, specialist_text):
    header, rows = parse(parent_text)
    if len(header) != 13 or not set(TARGETS) <= set(header):
        raise ValueError('Parent must have twelve findings')
    _, bag = parse(specialist_text, ['StudyInstanceUID', *TARGETS])
    lookup = {row[0]: row for row in bag}
    if set(lookup) != {row[0] for row in rows}:
        raise ValueError('Study coverage mismatch')
    result = [row.copy() for row in rows]
    for bag_index, label in enumerate(TARGETS, 1):
        column = header.index(label)
        parent_rank = ranks([float(row[column]) for row in rows])
        bag_rank = ranks([float(lookup[row[0]][bag_index]) for row in rows])
        for i, row in enumerate(result):
            row[column] = format(.9 * parent_rank[i] + .1 * bag_rank[i], '.17g')
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(result)
    return stream.getvalue()
