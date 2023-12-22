from config import DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR
from config import VISUALIZE_TRANSFORMED_IMAGES
from config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from model import create_model
from ultis import Averager
from tqdm.auto import tqdm
from datasets import train_dataset, train_loader, valid_loader

import torch
import matplotlib.pyplot as plt
import time

plt.style.use("ggplot")

# function for running training iterations


def train(train_data_loader, model):
    print("Training")
    global train_itr
    global train_loss_list
    train_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    for i, data in enumerate(prog_bar):
        # Zero out the gradients
        optimizer.zero_grad()

        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        # Backpropagation and optimize
        losses.backward()
        optimizer.step()

        train_itr += 1

        # update the loss value beside the progress bar for each interation
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        if loss_value < 0.05:
            break
    return train_loss_list


def validate(valid_data_loader, model):
    print("Validating")
    global val_itr
    global val_loss_list
    val_loss_list = []

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)

        val_loss_hist.send(loss_value)

        val_itr += 1

        # update the loss value beside the progress bar for each interation
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    return val_loss_list


def save_chart(out_dir, train_loss, val_loss, epoch):
    # create two subplots, one for each, training and validation
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color="blue")
    train_ax.set_xlabel("iterations")
    train_ax.set_ylabel("train loss")
    valid_ax.plot(val_loss, color="red")
    valid_ax.set_xlabel("iterations")
    valid_ax.set_ylabel("validation loss")
    figure_1.savefig(f"{out_dir}/{epoch}_train_loss.png")
    figure_2.savefig(f"{out_dir}/{epoch}_valid_loss.png")


if __name__ == "__main__":
    # initialize the model and over the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameter
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=0.0001, momentum=0.9, weight_decay=0.0005)

    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all
    # iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    # name to save the trained model with
    MODEL_NAME = "model"

    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from ultis import show_tranformed_image

        show_tranformed_image(train_loader)
    for epoch in range(NUM_EPOCHS):
        epoch_real = epoch + 1

        print(f"\nEPOCH {epoch_real} of {NUM_EPOCHS}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        # model.train()
        train_loss = train(train_loader, model)
        # model.eval()
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took{((end-start)/60):.3f} minutes for epoch {epoch}")

        if epoch_real % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
            torch.save(model.state_dict(), f"{OUT_DIR}/model_{epoch+1}.pth")
            print("SAVING MODEL COMPLETE...\n")

        if (
            epoch_real % SAVE_PLOTS_EPOCH == 0
        ):  # save loss plots and model once at the end
            save_chart(OUT_DIR, train_loss, val_loss, epoch_real)
            print("SAVING PLOTS COMPLETE")

        if epoch_real == NUM_EPOCHS:  # save loss plots and model once at the end
            save_chart(OUT_DIR, train_loss, val_loss, epoch_real)
            torch.save(model.state_dict(), f"{OUT_DIR}/model_{epoch+1}.pth")
            print("SAVING MODEL COMPLETE...\n")

        if float(train_loss_hist.value) < 0.05 or float(val_loss_hist.value) < 0.05:
            torch.save(model.state_dict(), f"{OUT_DIR}/best_model_{epoch+1}.pth")
            break
        plt.close("all")
        # sleep for 5 seconds after each epoch
        time.sleep(5)
