- there are 3 python files. : train, predict, script

- train is for training models. This need not be run. 

- For `predict.py` : make a folder called "val" in the same directory as predict.py, and put the test folder inside "val".
the run predict.py with python3

- `script.py` is used to generate output.txt file. This needs a text file "images.txt", which is created by doing: `ls > images.txt` in test_images folder. 

- Most of our code is inspired from PyTorch Tutorials, so we did not change much of it to avoid debugging it ourselves.

- the architecture image is a general image of Resnet18 taken from the internet. The last layer is a fully connected layer with 36 output classes.

