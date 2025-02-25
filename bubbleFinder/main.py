import torch
import torchvision
from functions.engine import train_one_epoch, evaluate
from functions import utils as utils
import time

from functions import bubble

def main():

    print("Pytorch version " + torch.__version__)
    print("torchvision version " + torchvision.__version__)

    #pathName = sys.argv[1]
    #print("Doing " + pathName + "...")
    pathName = './data/30images_60bubbles_20minDistance_95aperture'

    train = True

    if train:

        print('training...')

        t = time.time()
        # train on the GPU or on the CPU, if a GPU is not available
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # our dataset has two classes only - background and bubble
        num_classes = 2
        # use our dataset and defined transformations
        dataset = bubble.BubbleDataset(pathName, bubble.get_transform(train=True))
        dataset_test = bubble.BubbleDataset(pathName, bubble.get_transform(train=False))

        # split the dataset in train and test set
        indices = torch.randperm(len(dataset)).tolist()
        trainPerct = 0.8
        dataset_train = torch.utils.data.Subset(dataset, indices[:int(trainPerct * len(dataset))])

        dataset_test = torch.utils.data.Subset(dataset_test, indices[int(trainPerct * len(dataset)) + 1:])


        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset_train, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        # get the model using our helper function
        model = bubble.get_model_instance_segmentation(num_classes)

        # move model to the right device
        model.to(device)

        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # let's train it for 10 epochs
        num_epochs = 1

        for epoch in range(num_epochs):

            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # update the learning rate
            lr_scheduler.step()
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)

        torch.save(model, pathName + '/model.pt')

        dt = time.time() - t
        print("total training time: " + format(dt, ".0f") + " seconds")

if __name__ == "__main__":
    main()
