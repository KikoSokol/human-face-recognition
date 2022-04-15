def get_pairs_from_csv(file_name):
    pairs = []

    file = open(file_name, "r")

    lines = file.readlines()

    for line in lines:
        split_line = line.strip().split(",")
        pairs.append((split_line[0], split_line[1]))

    pairs.pop(0)
    return pairs
