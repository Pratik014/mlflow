import argparse

if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--n_estimators','-n',default=100,type=float)
    arg.add_argument('--max_depth','-m',default=5,type=int)
    arg.add_argument('--min_samples_split','-min',default=5,type=int)
    arg.add_argument('--max_samples','-max',default=5.0,type=float)
    parser_agr=arg.parse_args()

    print(parser_agr.n_estimators,parser_agr.max_depth,parser_agr.min_samples,parser_agr.max_sample)