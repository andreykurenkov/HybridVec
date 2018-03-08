from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--vocab-file", type=str, default="data/glove.6B.100d.txt")
  args = parser.parse_args()

  rare_words = [line.split()[0] for line in open("data/rw/rw.txt", "r").readlines()]
  rare_words = set(rare_words)

  train_output = open("data/train_{}".format(args.vocab_file), "w")
  test_output = open("data/test_{}".format(args.vocab_file), "w")

  for line in tqdm(open(args.vocab_file, "r"), total=4*10**5):
    word = line.split()[0]
    if word not in rare_words:
      train_output.write(line)
    else:
      test_output.write(line)
