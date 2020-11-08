import os
from pickX.utils.Toolbox import start_toolbox

USERSPACE_PATH = os.path.expanduser("~")

"""
dependencies versions:
python 3.7.8
tensorflow 2.2.0
tensorboard 2.2.2
"""


def main():
    if "pickX" not in os.listdir(USERSPACE_PATH):  # setup done?
        raise LookupError("couldn't find folder \"pickX\" at:{}. Please run setup.py first\n".format(USERSPACE_PATH))
    start_toolbox()


if __name__ == "__main__":
    main()
# D:/Richard's Laptop/git/pickx/sampledata/20200619/DS_TEST2
