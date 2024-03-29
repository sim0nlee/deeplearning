import torch

import models


def test(dataloader, model, loss_fn, epoch, device, shallow=False, writer=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.eval()

    test_loss, correct = 0, 0
    test_loss_shallow, correct_shallow = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if shallow:
                pred_shallow = model(X, shallow=True)
                test_loss_shallow += loss_fn(pred_shallow, y).item()
                correct_shallow += (pred_shallow.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n\n")
    if writer:
        writer.add_scalar('Test Accuracy', correct, epoch)

    if shallow:
        test_loss_shallow /= num_batches
        correct_shallow /= size
        print(f"Test Error Shallow: \n Accuracy: {(100 * correct_shallow):>0.1f}%, Avg loss: {test_loss_shallow:>8f} \n\n")
        if writer:
            writer.add_scalar('Test Accuracy Shallow', correct_shallow, epoch)

