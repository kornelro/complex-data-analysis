import csv
import sys

import numpy as np


def detect(
    in_file_path: str,
    out_file_path: str,
    window_size: int,
    verbose: bool = True
):
    # method assumes that stream data comes from normal distrbution
    # and estimates normal distribution parameters every data window
    # with given size

    if not isinstance(in_file_path, str):
        raise TypeError('Input file path must be string!')

    if not isinstance(out_file_path, str):
        raise TypeError('Output file path must be string!')

    if not isinstance(window_size, int):
        raise ValueError('Reference set size must be int!')
    if window_size <= 0:
        raise ValueError('Reference set size must be grater than 0!')

    try:
        verbose = bool(verbose)
    except TypeError:
        raise TypeError(
            'Verbose must be convertable to bool!'
        )

    with open(in_file_path, 'r') as f_in, open(out_file_path, 'w') as f_out:

        reader = csv.reader(f_in)
        # skip header
        next(reader)

        writer = csv.writer(f_out)
        # write header
        writer.writerow(['window_num', 'mu', 'sigma'])

        window_data = []
        window_num = 1
        for val in reader:
            window_data.append(float(val[0]))
            if len(window_data) == window_size:
                window_data = np.array(window_data)
                loc = np.mean(window_data)
                scale = np.sqrt(np.var(window_data))
                writer.writerow(
                    [
                        window_num,
                        loc,
                        scale
                    ]
                )
                if verbose:
                    print(''.join(
                        [
                            'Window num: ',
                            str(window_num),
                            ' Loc: ',
                            str(loc),
                            ' Scale: ',
                            str(scale)
                        ]
                    ))
                window_data = []
                window_num += 1


if __name__ == '__main__':

    in_file_path = sys.argv[1]
    out_file_path = sys.argv[2]
    window_size = int(sys.argv[3])
    try:
        verbose = bool(int(sys.argv[4]))
    except KeyError:
        verbose = True

    detect(
        in_file_path=in_file_path,
        out_file_path=out_file_path,
        window_size=window_size,
        verbose=verbose
    )
