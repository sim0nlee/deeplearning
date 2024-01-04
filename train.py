import torch

from activation import TReLU


def train(dataloader, batch_size, model, loss_fn, optimizer, epoch, device, writer=None):

    size    = len(dataloader.dataset)
    n_steps = len(dataloader)

    model.train()

    for i, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print_model_data(model, n_steps, epoch, i, writer)

        optimizer.zero_grad()

        if i % (2**5) == 0:
            write_loss(loss, n_steps, batch_size, size, epoch, i, writer)


def print_model_data(model, n_steps, cur_epoch, cur_iter, writer):

    weight_grads = []
    bias_grads   = []
    alphas       = []

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
        writer.add_scalar('Weights Gradient Norms', weight_grad_norms.item(), n_steps * cur_epoch + cur_iter)
        writer.add_scalar('Bias Gradient Norms', bias_grad_norms.item(), n_steps * cur_epoch + cur_iter)
    print('TReLU alphas:', ["{:.5f}".format(alpha) for alpha in alphas])
    print()


def write_loss(loss, n_steps, batch_size, size, cur_epoch, cur_iter, writer=None):
    loss, current = loss.item(), (cur_iter + 1) * batch_size
    if writer:
        writer.add_scalar('Training Loss', loss, n_steps * cur_epoch + cur_iter)
    print(f"Loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
