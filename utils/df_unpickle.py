## Program to convert a pickled dataframe to a tsv file

import pandas as pd
import argparse

def parse_args():
    parser=argparse.ArgumentParser(description="convert a pickled dataframe to a tsv file")
    parser.add_argument("--in", help="input pickle file", dest="in_file", type=str)
    parser.add_argument("--out", help="output tsv file", dest="out_file", type=str)
    parser.add_argument("--format", help="output format", dest="format", type=str, default="auto")
    parser.add_argument("--verbose", help="verbose output", dest="verbose", type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_args()
    df = pd.read_pickle(args.in_file)

    if args.format == "auto":
        ending = args.out_file.split(".")[-1]
        if ending in ["tsv", "txt"]:
            args.format = "tsv"
        elif ending in ["csv"]:
            args.format = "csv"
        elif ending in ["xls", "xlsx"]:
            args.format = "excel"
        else:
            raise ValueError("Unknown file format: {}".format(ending))

    if args.format == "tsv":
        df.to_csv(args.out_file, sep="\t", index=True)
    elif args.format == "csv":
        df.to_csv(args.out_file, index=True)
    elif args.format == "excel":
        df.to_excel(args.out_file, index=True)
    else:
        raise ValueError("Unknown file format: {}".format(args.format))

    if args.verbose:
        print(df.head())

    return None

if __name__=="__main__":
    main()

