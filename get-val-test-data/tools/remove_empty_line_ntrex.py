import argparse

def main(args):
    ntrex_dir = args.ntrex_dir
    l=[]
    with open('{}/newstest2019-ref.urd.txt'.format(ntrex_dir), 'r') as f:
        for line in f:
            if line != '\n':
                l.append(line.strip())


    with open('{}/newstest2019-ref.urd.txt'.format(ntrex_dir), 'w') as fp:
        for item in l:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ntrex-dir', type=str, required=True)
    args = parser.parse_args()

    main(args)