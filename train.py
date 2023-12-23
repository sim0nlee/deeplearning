import torch

from activation import TReLU


def train(dataloader, model, loss_fn, optimizer, epoch, device, writer=None):
    size = len(dataloader.dataset)
    model.train()

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            write_grads_and_alphas(model, size, epoch, i, writer)

        optimizer.zero_grad()

        if i % (2**5) == 0:
            write_loss(loss, size, epoch, i, writer)


def write_grads_and_alphas(model, size, cur_epoch, cur_iter, writer):
    weight_grads = []
    bias_grads = []
    alphas = []
    for module in model.net:
        if isinstance(module, torch.nn.Linear):
            weight_grads.append(module.weight.grad.ravel())
            bias_grads.append(module.bias.grad.ravel())
        if isinstance(module, TReLU):
            for param in module.parameters():
                alphas.append(param.data.item())

    weight_grad_norms = torch.linalg.norm(torch.cat(weight_grads))
    bias_grad_norms = torch.linalg.norm(torch.cat(bias_grads))
    print()
    print('Weight grads norm:', weight_grad_norms.item())
    print('Bias grads norm:', bias_grad_norms.item())
    if writer:
        writer.add_scalar('Weights Gradient Norms', weight_grad_norms.item(), size * cur_epoch + cur_iter)
        writer.add_scalar('Bias Gradient Norms', bias_grad_norms.item(), size * cur_epoch + cur_iter)
    print('TReLU alphas:', ["{:.5f}".format(alpha) for alpha in alphas])
    print()


def write_loss(loss, size, cur_epoch, cur_iter, writer=None):
    loss, current = loss.item(), (cur_iter + 1) * size
    if writer:
        writer.add_scalar('Training Loss', loss, size * cur_epoch + cur_iter)
    print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
