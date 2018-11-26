from argparse import ArgumentParser
from tqdm import tqdm


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--vocab-file", type=str, default="data/glove/glove.6B.100d.txt")
  args = parser.parse_args()

  rare_words = [line.split()[0] for line in open("data/rw/rw.txt", "r").readlines()]
  rare_words = set(rare_words)

  val_words = [line.split()[0] for line in open("data/rw/val.txt", "r").readlines()]
  val_words = set(val_words) - rare_words

  train_output = open("data/glove/train_{}".format(args.vocab_file.split("/")[-1]), "w")
  val_output = open("data/glove/val_{}".format(args.vocab_file.split("/")[-1]), "w")
  test_output = open("data/glove/test_{}".format(args.vocab_file.split("/")[-1]), "w")

  for line in tqdm(open(args.vocab_file, "r"), total=4*10**5):
    word = line.split()[0]
    if word in rare_words:
      test_output.write(line)
    elif word in val_words:
      val_output.write(line)
    else:
      train_output.write(line)
