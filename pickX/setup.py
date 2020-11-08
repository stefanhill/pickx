import os
import datetime
from os.path import expanduser
from pickX.data.DataHandler import DataHandler
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from pickX.ml.CNN import CNN


def main():
    USERSPACE_PATH = os.path.join(expanduser("~"), "pickX")
    ans = input("started setup! this setup is just for training the default classifier once,\n"
                "if you continue, any default classifier may be overwritten! continue? y/n: ")

    if ans.lower() != 'y':
        quit()

    print("{} training first classifier, this may take longer, depending on your machine...".format(
        datetime.datetime.now()))

    if not os.path.isdir(USERSPACE_PATH):
        os.mkdir(USERSPACE_PATH)

    if "models" not in os.listdir(os.path.join(USERSPACE_PATH)):
        os.mkdir(os.path.join(USERSPACE_PATH, "models"))
        os.mkdir(os.path.join(USERSPACE_PATH, "logs"))
        os.mkdir(os.path.join(USERSPACE_PATH, "checkpoints"))

    features = DataHandler.unpickle("features_all_projects", "test\\python")
    labels = DataHandler.unpickle("labels_all_projects", "test\\python")

    NAME = "default"
    input_shape = features.shape[1:]
    cnn = CNN()
    cnn.initialize([[64, 7], [64, 5]], 2, 32, input_shape)
    model = cnn.model
    tensorboard = TensorBoard(log_dir=USERSPACE_PATH + "\\logs\\{}.log".format(NAME))
    checkpoint = ModelCheckpoint(filepath=USERSPACE_PATH + "\\checkpoints\\{}.checkpoint".format(NAME),
                                 save_weights_only=True, verbose=1)

    model.fit(x=features, y=labels, epochs=1, batch_size=32, validation_split=0.1, callbacks=[tensorboard, checkpoint])
    cnn.__export__(USERSPACE_PATH + "\\models\\{}.model".format(NAME))
    model.summary()

    print("setup complete!\n")
    print("The models and their corresponding logs/checkpoints will be saved under {}\n".format(expanduser("~")))


if __name__ == '__main__':
    main()
