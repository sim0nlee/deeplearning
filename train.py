import os
import torch
from tqdm import tqdm

from resnet import Bottleneck
from activation import TReLU


def train(train_loader, model, num_epochs, epoch, device, optimizer, criterion, train_writer, checkpoint_interval,
          checkpoint_dir):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for step, (inputs, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)

        # Log training params (alpha, beta, grads) to TensorBoard
        if step % 100 == 0:
            write_params(model, train_writer, train_loader, epoch, step)

        # Log training loss and accuracy to TensorBoard
        if step % (2**5) == 0:
            write_loss(train_writer, loss, epoch, train_loader, step, correct_predictions, total_samples)

    average_loss = running_loss / len(train_loader)
    accuracy = correct_predictions / total_samples
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {average_loss}, Training Accuracy: {accuracy}")

    # Save checkpoint
    if (epoch + 1) % checkpoint_interval == 0:
        checkpoint_path = os.path.join(checkpoint_dir, 'resnet_model.pth')  # Overwrite the same file
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

def write_params(model, train_writer, train_loader, epoch, step):

    # Adding the gradients norm and alphas to TensorBoard
    weight_grads = []
    bias_grads = []
    alphas = []
    for module in model.net:
        if isinstance(module, Bottleneck):
            if module.is_Bottleneck:
                weight_grads.append(module.conv1_1x1.weight.grad.ravel())
                bias_grads.append(module.conv1_1x1.bias.grad.ravel())
                weight_grads.append(module.conv2_3x3.weight.grad.ravel())
                bias_grads.append(module.conv2_3x3.bias.grad.ravel())
                weight_grads.append(module.conv3_1x1.weight.grad.ravel())
                bias_grads.append(module.conv3_1x1.bias.grad.ravel())
            else:
                weight_grads.append(module.conv1_3x3.weight.grad.ravel())
                bias_grads.append(module.conv1_3x3.bias.grad.ravel())
                weight_grads.append(module.conv2_3x3.weight.grad.ravel())
                bias_grads.append(module.conv2_3x3.bias.grad.ravel())
            if isinstance(module.activation, TReLU):
                alphas.append(module.trelu_params()[0].data.item())
        elif isinstance(module, torch.nn.Linear):
            weight_grads.append(module.weight.grad.ravel())
            bias_grads.append(module.bias.grad.ravel())
        elif isinstance(module, TReLU):
            for param in module.parameters():
                alphas.append(param.data.item())

    weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
    bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
    train_writer.add_scalar('Weights Gradient Norms', weight_grad_norms.item(), len(train_loader) * epoch + step)
    train_writer.add_scalar('Bias Gradient Norms', bias_grad_norms.item(), len(train_loader) * epoch + step)

    # Adding the beta parameters to TensorBoard
    if model.beta is not None:
        if model.beta_is_global:
            train_writer.add_scalar('Global Beta', model.beta.item(), len(train_loader) * epoch + step)
        else:
            for i, b in enumerate(model.beta):
                train_writer.add_scalar(f'Betas/Beta_{i}', b.item(), len(train_loader) * epoch + step)


def write_loss(train_writer, loss, epoch, train_loader, step, correct_predictions, total_samples):
    train_writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + step)
    train_writer.add_scalar('Training Accuracy', correct_predictions / total_samples,
                            epoch * len(train_loader) + step)
