from argparse import ArgumentParser, SUPPRESS
from sys import argv
import pandas as pd
from DecisionTree import DecisionTreeClassifier

def display_help():
    print('\nPlease specify one of the following datasets:')
    print("- iris")
    print("- restaurant")
    print("- weather")
    print("- connect4\n")

    print("To add a new dataset for testing, place it in the datasets folder")
    print("and run the program again with the dataset name as an argument.")
    print("Example:\n")

    print("python3 main.py [-tr,--train] 'dataset'\n")

    print("To test a previously trained tree, run the program as follows:\n")

    print("python3 main.py [-tr,--train] 'dataset' [-t,--test] 'test_dataset'\n")

    exit(0)

if __name__ == '__main__':
    if len(argv) == 2:
        if argv[1] == '-h' or argv[1] == '--help':
            display_help()
            exit()

    parser = ArgumentParser(add_help=False, description='Decision Tree Classifier Program')

    parser.add_argument('-h', '--help', default=SUPPRESS, help='Show the help menu')
    parser.add_argument('-tr', '--train', default=SUPPRESS, help='Dataset to be used to train the Decision Tree')
    parser.add_argument('-t', '--test', default=SUPPRESS, help='Dataset to be used to test the Decision Tree')

    args = parser.parse_args()

    if len(argv) == 1:
        parser.print_help()
        exit()

    try:
        if args.train is not None:
            dataset_path = "datasets/" + str(args.train) + ".csv"
            try:
                data_frame = pd.read_csv(dataset_path, na_values=['NaN'], keep_default_na=False)
            except FileNotFoundError:
                print("The dataset '" + str(args.train) + "' was not found.")
                exit()
            else:
                print("Decision Tree Classifier for the '" + str(args.train) + "' dataset:\n")

                classifier = DecisionTreeClassifier(data_frame)

                # Print the DataFrame and the Decision Tree
                # print(data_frame)
                print(classifier)

            try:
                if args.test is not None:
                    test_path = "datasets/" + str(args.test) + ".csv"
                    try:
                        test_data = pd.read_csv(test_path, na_values=['NaN'], keep_default_na=False)
                    except FileNotFoundError:
                        print("The dataset '" + str(args.test) + "' was not found.")
                    else:
                        print("Predicted values for the '" + str(args.test) + "' dataset:\n")

                        print(classifier.classify(test_data))
            except AttributeError:
                pass
    except AttributeError:
        print("\nPlease provide a dataset to train the Decision Tree Classifier.")
        print("You can view available datasets by passing '-h' or '--help' as an argument.")
