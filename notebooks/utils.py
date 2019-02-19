import csv
def isfloat(x):
    try:
        a = float(x)
    except ValueError:
        return False
    else:
        return True

def isint(x):
    try:
        a = float(x)
        b = int(a) if a != float('Inf') else 9999999
    except ValueError:
        return False
    else:
        return a == b

def read_results(filename):
    results = {}
    headers = []
    with open(filename, 'r') as csvfile:
        resultsreader = csv.reader(csvfile)
        for i, row in enumerate(resultsreader):
            if i == 0:
                for col in row:
                    results[col] = []
                    headers.append(col)  # to keep track of an ordered addition of the headings
            else:
                for i, col in enumerate(row):
                    results[headers[i]].append(int(float(col)) if isint(col) else float(col))
    return results	
