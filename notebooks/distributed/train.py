from tqdm import trange
import torch

def train(model, optimizer, loss_function, epochs, device, trainloader, validloader):

    model.to(device)
    model.train()

    train_losses, valid_losses, accuracies = [], [], []

    pbar = trange(epochs, desc='Training', leave=True)
    for epoch in pbar:
        epoch_loss = 0

        # Training loop for one epoch
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = loss_function(output, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # Evaluation
        valid_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                valid_loss += loss_function(output, labels).item()
                p = torch.exp(output)
                top_p, top_c = p.topk(1, dim=1)
                equals = (top_c == labels.view(*top_c.shape)).type(torch.FloatTensor)
                accuracy += torch.mean(equals)

        model.train()

        train_losses.append(epoch_loss/len(trainloader))
        valid_losses.append(valid_loss/len(validloader))
        accuracies.append(accuracy.item()/len(validloader)*100)
        pbar.set_postfix({"Accuracy": accuracies[-1]})

    return train_losses, valid_losses, accuracies


#def train(model, optimizer, loss_function, epochs, device):
#    import time
#
#    # Ensure the model is running on the device
#    model.to(device)
#
#    train_losses, valid_losses, accuracies = [], [], []
#
#    pbar = trange(epochs, desc='Training', leave=True)
#    for epoch in pbar:
#
#        epoch_loss = 0
#
#        # Train the model for one epoch
#        for images, labels in trainloader:
#
#            # Move data to GPU
#            images, labels = images.to(device), labels.to(device)
#
#            # Reset optimizer gradients to zero
#            optimizer.zero_grad()
#
#            # Perform forward pass
#            output = model(images)
#
#            # Compute the loss
#            loss = loss_function(output, labels)
#
#            # Perform backpropagation
#            loss.backward()
#
#            # Update model weights
#            optimizer.step()
#
#            epoch_loss += loss.item()
#
#        # Evalutate the model on the validation set
#        valid_loss = 0
#        accuracy = 0
#
#        with torch.no_grad(): 
#            for images, labels in validloader:
#
#                # Move data to GPU
#                images, labels = images.to(device), labels.to(device)
#
#                # Perform forward pass
#                output = model(images)
#
#                # Accumulate the validation loss
#                # Hint: use item() to convert a single-entry tensor into a number
#                valid_loss += loss_function(output, labels).item() 
#
#                # Compute class probabilities
#                p = torch.exp(output)
#
#                # Compute accuracy
#                top_p, top_c = p.topk(1, dim=1) # Top prediction
#                equals = (top_c == labels.view(*top_c.shape)).type(torch.FloatTensor)
#                accuracy += torch.mean(equals)
#
#        train_losses.append(epoch_loss/len(trainloader))
#        valid_losses.append(valid_loss/len(validloader))
#        accuracies.append(accuracy.item()/len(validloader)*100)
#
#        pbar.set_postfix({"Accuracy": accuracies[-1]})
#
#    return train_losses, valid_losses, accuracies
#
