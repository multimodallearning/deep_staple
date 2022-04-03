import sys
import argparse

if __name__ == "__main__":
    main(sys.argv[1:])

def main(argv):
    parser.add_argument("-i", "--input-folder", required=True)
    args = parser.parse_args(argv)
