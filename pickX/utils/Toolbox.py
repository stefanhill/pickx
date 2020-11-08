import os
from pickX.utils.ExecutionHandler import ExecutionHandler

USERSPACE_PATH = os.path.join(os.path.expanduser("~"), "pickX")


def predict():
    """
    Commandline tool to predict picks
    :return: void
    """
    model_name = None
    ans = input("do you want to predict with the current default model? [3]\n"
                "do you want to choose another model to train with? [4]\n")
    if ans == '3':
        model_name = 'default'
    elif ans == '4':
        choice = str(input("available models: \n"
                           "{}\n"
                           "choose any of the above: ".format(
            [x[:x.rindex(".")] for x in os.listdir(USERSPACE_PATH + "\\models")])))
        if choice + '.model' in os.listdir(USERSPACE_PATH + "\\models"):
            model_name = choice
        else:
            quit("model was not found")
    else:
        quit()
    # model is chosen
    path = input("input root path to project pickled to start the prediction\n"
                 "example: C:\\your_path_to_project\\\n")
    if not os.path.isdir(path):
        quit("{} is not a correct pickled path".format(path))

    print(ExecutionHandler.predict(model_name, path))


def train():
    """
    Commandline tool to train a model
    :return: void
    """
    model_conf = None
    model_type = str(input("train new model with \n"
                           "express mode[1]\n"
                           "expert mode[2]\n"))

    if model_type == '1':
        print("A large model will take longer to train, but yields better results.\n"
              "A smaller model will take shorter time to train, but will yield worse results")

        d = {
            "large": (([[512, 7], [256, 5], [256, 3]], 3, 64), 10),
            "medium": (([[256, 7], [128, 5], [128, 3]], 2, 64), 10),
            "small": (([[128, 5], [64, 3]], 2, 32), 10),
        }
        d_key = str(input("choose from types below:\n"
                          "{}\n".format(list(d.keys()))))
        if d_key not in d.keys():
            quit()
        model_conf = d[d_key]

    elif model_type == '2':
        print("please input params for convolutional layers as comma-separated list:\n"
              "eg. 256 5,128 3,...\n")
        layers = [list(map(int, x.split(" "))) for x in input("conv layers: \n").split(",")]
        dense_layers = int(input("number of dense layers: \n"))
        dense_nodes = int(input("number of neurons per dense layer: \n"))
        epochs = int(input("epochs: \n"))
        model_conf = ((layers, dense_layers, dense_nodes), epochs)

    else:
        quit()
    path = str(input("input root path to project pickled with training data:\n"))
    while True:
        model_name = input("input model name (naming your model \"default\" is not allowed!): \n")
        if model_name != "default":
            break

    if not os.path.isdir(path):
        quit("couldn't find pickled at path {}".format(path))

    ExecutionHandler.train(model_name, path, model_conf)


def validate():
    """
    Commandline tool to validate a given data set on a model
    :return: void
    """
    model_name = str(input("available models: \n"
                           "{}\n"
                           "choose any of the above: ".format(
        [x[:x.rindex(".")] for x in os.listdir(os.path.join(USERSPACE_PATH, "models"))])))
    validation_path = str(input("please input the path to the dataset you want to use for validation.\n"))
    if model_name + '.model' in os.listdir(os.path.join(USERSPACE_PATH, "models")):
        res = ExecutionHandler.test(model_name, validation_path)
        print(res)
    else:
        print("couldn't find model")


def convert():
    """
    Commandline tool to convert a data set to a pickled training set
    :return: void
    """
    dataset_path = str(input("please input the path to the dataset you want to use for validation.\n"))
    conversion = ExecutionHandler.prepare_dataset_for_training(dataset_path)
    if conversion:
        print("Conversion successful.")
    else:
        raise ValueError(
            "File cannot be converted. Please check the path or have a look whether the dataset is already converted.")


def start_toolbox():
    """
    main handler to start the commandline tools
    :return: void
    """
    ans = str(input("Do you want to predict picks? [1]\n"
                    "Do you want to train a new Classifier? [2]\n"
                    "Do you want to validate a model? [3]\n"
                    "Do you want to convert a dataset into pickle format? [4]\n"))

    if ans == '1':
        predict()

    elif ans == '2':
        train()

    elif ans == '3':
        validate()

    elif ans == '4':
        convert()

    else:
        quit()
