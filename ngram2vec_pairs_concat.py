import argparse

def main():
    #CLI arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input_path", type=str, default=r'../Models/Ngram2vec_model/', help="Path to the folder where pair files are")
    parser.add_argument("--window", type=int, default=5, help="Window size. default=5")
    args = parser.parse_args()

    files = ['pairs_%d_0' % args.window, 'pairs_%d_1' % args.window, 'pairs_%d_2' % args.window, 'pairs_%d_3' % args.window]
    output_path = args.input_path + "pairs_%d.txt" % args.window

    output_file = open(output_path, "w", encoding="utf-8")
    output_file.write("")
    output_file.close()
    print("Cleared concatenated output pairs text file")

    output_file = open(output_path, "a", encoding="utf-8")

    for pairs_file in files:
        with open(args.input_path + pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                output_file.write(line)
            print('%s appended...' % pairs_file)


    output_file.close()
    print('Done! Concatinated ngram2vec pairs files.')

if __name__ == '__main__':
    main()