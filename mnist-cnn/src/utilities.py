import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import os

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Train Function
def train_model(model, dataloader, cost_function, optimizer, EPOCHS):
    # set the model to training mode
    model.train()
    losses = []
    accuracies = []

    # run loop for epoch
    for epoch in range(EPOCHS):
        # print("="*50)
        print(f"EPOCH: {epoch+1}...")
        # print()
        total_loss = correct_pred = total_pred = 0

        # for each batch of image, label pairs
        for images, labels in dataloader:
            # Transfer to `device`
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)
            # Clear gradients (from previous steps)
            optimizer.zero_grad()
            # compute cost function
            loss = cost_function(outputs, labels)
            # Backpropogate
            loss.backward()
            optimizer.step()
            # Calculate total loss
            total_loss = loss.item()*images.size(0) # because, loss returned would be average
            # Calculate counts
            predicted = torch.argmax(outputs, dim=1)
            correct_pred += (predicted == labels).sum().item()
            total_pred += labels.size(0)
        # Compute metrics for each EPOCH
        epoch_accuracy = correct_pred/total_pred
        
        losses.append(total_loss/total_pred)
        accuracies.append(epoch_accuracy)

        # print the metrics
        
        # print(f"Total EPOCH Loss: {total_loss}")
        # print(f"EPOCH Accuracy: {epoch_accuracy:.4f}")
    return losses, accuracies

# define an evaluation function
def evaluate(model, dataloader, loss_function):
    # set model to evaluate mode
    model.eval()
    
    test_loss = correct_pred = total_pred = 0

    # disable gradients
    with torch.no_grad():
        for images, labels in dataloader:
            # move to `device`
            images, labels = images.to(device), labels.to(device)

            # make predictions
            outputs = model(images)

            # loss
            loss = loss_function(outputs, labels)
            # total loss
            test_loss += loss.item()*images.size(0)

            # extract a prediction for labels
            predicted = torch.argmax(outputs, dim=1)

            # calculate total correct predictions
            correct_pred += (predicted == labels).sum().item()
            total_pred += labels.size(0)

        # compute metrics
        test_accuracy = correct_pred/total_pred
        return test_loss/total_pred, test_accuracy

# visualize using plots
def plot_metrics(losses, accuracies,model_name):
    # plt.figure(figsize=(16,9))
    fig,axes = plt.subplots(1,2,figsize=(16,9))
    epochs = range(1,len(losses)+1)
    axes[0].plot(epochs, losses, marker="^")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].set_title("Total loss over Epochs",)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Total Loss")
    axes[0].set_xticks(epochs)

    axes[1].plot(epochs, accuracies, marker="^")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xticks(epochs)
    
    # Optional: sanitize model_name for filesystem safety
    safe_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', str(model_name)).strip('_')
    out_dir = "../training-plots"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{safe_name}.png")
    
    plt.suptitle(f"Model: {model_name}",fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.show()

# now visualize the prediction
def prediction_and_image(model, dataloader):
    # select an image randomly
    test_image, test_label = dataloader[np.random.choice(10000)]
    test_image = test_image.to(device).unsqueeze(0)

    # put the model in evaluation (inference) mode
    model.eval()
    predictions_test_image = model(test_image)
    probabilities = torch.nn.functional.softmax(predictions_test_image, dim=1)
    probabilities = probabilities.cpu().detach().numpy() #.cpu() to copy tensor to memory first


    plt.Figure(figsize=(20,20))

    plt.subplot(1,2,1)
    classes = [i for i in range(10)]
    x_values = classes
    x_labels = [str(i) for i in x_values]
    plt.bar([i for i in range(10)], probabilities[0])
    plt.xticks(x_values, x_labels)

    plt.subplot(1,2,2)
    plt.imshow(test_image[0,0,:,:].cpu(), cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])

    model_output = np.argmax(probabilities)
    print(f"The image is of the digit (as recognized by our model): {model_output}")
    print(f"The image is labelled as: {test_label} in the dataset.")

# print the results
def print_conclusion(model_name, losses, accuracies, test_loss, test_accuracy):
    print(f"Model: {model_name}")
    print(f"Loss(Training): {losses[-1]:.6f}")
    print(f"Accuracy(Training): {accuracies[-1]:.6f}")
    print(f"Loss(Testing): {test_loss:.6f}")
    print(f"Accuracy(Testing): {test_accuracy:.6f}")

def realworld_prediction(model, test_image_tensor):
    test_image_tensor = test_image_tensor.to(device).unsqueeze(0)

    # put the model in evaluation (inference) mode
    model.eval()
    predictions_test_image_tensor = model(test_image_tensor)
    probabilities = torch.nn.functional.softmax(predictions_test_image_tensor, dim=1)
    probabilities = probabilities.cpu().detach().numpy() #.cpu() to copy tensor to memory first


    plt.Figure(figsize=(20,20))

    plt.subplot(1,2,1)
    classes = [i for i in range(10)]
    x_values = classes
    x_labels = [str(i) for i in x_values]
    plt.bar([i for i in range(10)], probabilities[0])
    plt.xticks(x_values, x_labels)

    plt.subplot(1,2,2)
    plt.imshow(test_image_tensor[0,0,:,:].cpu(), cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])

    model_output = np.argmax(probabilities)
    print(f"The image is of the digit (as recognized by our model): {model_output}")