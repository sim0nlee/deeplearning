import torch
from tqdm import tqdm

def test(test_loader, model, num_epochs, epoch, device, test_writer):
    model.eval()
    correct_predictions_test = 0
    total_samples_test = 0

    with torch.no_grad():
        for inputs_test, labels_test in tqdm(test_loader, desc=f'Testing Epoch {epoch + 1}/{num_epochs}'):
            inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)

            outputs_test = model(inputs_test)
            _, predicted_test = torch.max(outputs_test, 1)

            correct_predictions_test += (predicted_test == labels_test).sum().item()
            total_samples_test += labels_test.size(0)

    accuracy_test = correct_predictions_test / total_samples_test

    # Log test accuracy to TensorBoard
    test_writer.add_scalar('Test Accuracy', accuracy_test, epoch + 1)

    print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy_test}")
