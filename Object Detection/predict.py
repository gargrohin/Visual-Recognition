import torch
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os


def predict_model(model):

    was_training = model.training
    model.eval()
    file1 = open("val_results.txt","w")


    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            for i in preds:
                file1.write(class_names[i.item()])
                file1.write("\n")

        file1.close()
        model.train(mode=was_training)




#fine_pred=["aircrafts@1", "aircrafts@2", "aircrafts@3", "aircrafts@4", "aircrafts@5", "aircrafts@6", "aircrafts@7", "birds@1", "birds@2", "birds@3", "birds@4", "birds@5", "birds@6", "birds@7", "birds@8", "birds@9", "birds@10", "birds@11","cars@1", "cars@2", "cars@3", "cars@4", "cars@5", "cars@6", "cars@7", "cars@8", "dogs@1", "dogs@2", "dogs@3", "dogs@4", "dogs@5", "flowers@1", "flowers@2", "flowers@3", "flowers@4", "flowers@5"]


data_transforms = {
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


data_dir = './'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in [ 'val']}
print(os.path.join(data_dir, 'val'))



dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=False, num_workers=4)
              for x in [ 'val']}



dataset_sizes = {x: len(image_datasets[x]) for x in ['val']}

class_names = image_datasets['val'].classes


device = "cpu"


model = torch.load("model1.pth")

predict_model(model)
