from datetime import datetime


def get_args_string(args):
    """
    Creates a string summarising the argparse arguments.
    :param args: parser.parse_args()

    :return: String of the arguments of the argparse namespace.
    """
    string = ""
    if hasattr(args, "experiment_name"):
        string += f"{args.experiment_name} ({datetime.now()})\n"
    max_length = max([len(k) for k, _ in vars(args).items()])
    new_dict = dict((k, v) for k, v in sorted(vars(args).items(), key=lambda x: x[0]))
    for key, value in new_dict.items():
        string += " " * (max_length - len(key)) + key + ": " + str(value) + "\n"
    return string


def yes_or_no(question) -> bool:
    reply = " "
    while reply[0] != "y" and reply[0] != "n":
        reply = str(input(question + " (y/n): ")).lower().strip()

    if reply[0] == "y":
        return True
    if reply[0] == "n":
        return False
