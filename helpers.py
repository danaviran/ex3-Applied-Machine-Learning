import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from models import Ridge_Regression
from models import Logistic_Regression


def plot_decision_boundaries(model, X, y, title='Decision Boundaries'):
    """
    Plots decision boundaries of a classifier and colors the space by the prediction of each point.
    Parameters:
    - model: The trained classifier (sklearn model).
    - X: Numpy Feature matrix.
    - y: Numpy array of Labels.
    - title: Title for the plot.
    """
    # h = .02  # Step size in the mesh
    # enumerate y
    y_map = {v: i for i, v in enumerate(np.unique(y))}
    enum_y = np.array([y_map[v] for v in y]).astype(int)
    h_x = (np.max(X[:, 0]) - np.min(X[:, 0])) / 200
    h_y = (np.max(X[:, 1]) - np.min(X[:, 1])) / 200
    # Plot the decision boundary.
    added_margin_x = h_x * 20
    added_margin_y = h_y * 20
    x_min, x_max = X[:, 0].min() - added_margin_x, X[:, 0].max() + added_margin_x
    y_min, y_max = X[:, 1].min() - added_margin_y, X[:, 1].max() + added_margin_y
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h_x), np.arange(y_min, y_max, h_y))
    # Make predictions on the meshgrid points.
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    print(Z.shape)
    Z = np.array([y_map[v] for v in Z])
    Z = Z.reshape(xx.shape)
    vmin = np.min([np.min(enum_y), np.min(Z)])
    vmax = np.min([np.max(enum_y), np.max(Z)])
    # Plot the decision boundary.
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8, vmin=vmin, vmax=vmax)
    # Scatter plot of the data points with matching colors.
    plt.scatter(X[:, 0], X[:, 1], c=enum_y, cmap=plt.cm.Paired, edgecolors='k', s=40, alpha=0.7,
                vmin=vmin, vmax=vmax)
    plt.title("Decision Boundaries")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(title)
    plt.show()


def read_data_demo(filename='train.csv'):
    """
    Read the data from the csv file and return the features and labels as numpy arrays.
    # the data in pandas dataframe format
    df = pd.read_csv(filename)
    # extract the column names
    col_names = list(df.columns)
    # the data in numpy array format
    data_numpy = df.values
    return data_numpy, col_names
    """
    # the data in pandas dataframe format
    df = pd.read_csv(filename)
    # Extract feature columns (assuming the last column is the label)
    X_train = df.iloc[:, :-1].values
    # Extract label column
    Y_train = df.iloc[:, -1].values
    return X_train, Y_train


def ridge_regression_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_validation, y_validation = read_data_demo('validation.csv')
    x_test, y_test = read_data_demo('test.csv')
    lambda_values = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    train_accuracies, test_accuracies, validation_accuracies = [], [], []
    for lambd in lambda_values:
        ridge_model = Ridge_Regression(lambd)
        ridge_model.fit(x_train, y_train)
        # Predictions and Accuracies
        train_predictions = ridge_model.predict(x_train)
        train_accuracies.append(np.mean(train_predictions == y_train))
        validation_predictions = ridge_model.predict(x_validation)
        validation_accuracies.append(np.mean(validation_predictions == y_validation))
        test_predictions = ridge_model.predict(x_test)
        test_accuracies.append(np.mean(test_predictions == y_test))
    # Plotting the accuracies
    plt.plot(lambda_values, train_accuracies, marker='o', linestyle='', label='Training Accuracy')
    plt.plot(lambda_values, validation_accuracies, marker='o', linestyle='',
             label='Validation Accuracy')
    plt.plot(lambda_values, test_accuracies, marker='o', linestyle='', label='Test Accuracy')
    # Display accuracies next to each dot
    for i, txt in enumerate(train_accuracies):
        plt.annotate(f'{txt:.3f}', (lambda_values[i], train_accuracies[i]), fontsize='x-small')
    for i, txt in enumerate(validation_accuracies):
        plt.annotate(f'{txt:.3f}', (lambda_values[i], validation_accuracies[i]), fontsize='x-small')
    for i, txt in enumerate(test_accuracies):
        plt.annotate(f'{txt:.3f}', (lambda_values[i], test_accuracies[i]), fontsize='x-small')
    plt.xlabel('Lambda')
    plt.ylabel('Accuracy')
    plt.title('Ridge Regression Model Performance')
    plt.legend()
    plt.show()
    # Plotting best model
    best_ridge_model = Ridge_Regression(2)
    best_ridge_model.fit(x_train, y_train)
    plot_decision_boundaries(best_ridge_model, x_test, y_test,
                             'Ridge Regression, lambda=2, accuracy: 0.97')
    # Plotting worst model
    worst_ridge_model = Ridge_Regression(10)
    worst_ridge_model.fit(x_train, y_train)
    plot_decision_boundaries(worst_ridge_model, x_test, y_test,
                             'Ridge Regression, lambda=10, accuracy: 0.944')


def given_function(x, y):
    return (x - 3) ** 2 + (y - 5) ** 2


def gradient(x, y):
    df_dx = 2 * (x - 3)
    df_dy = 2 * (y - 5)
    return np.array([df_dx, df_dy])


def gradient_descent_algorithm(iterations, learning_rate):
    x, y = 0, 0
    trajectory = []
    for i in range(iterations):
        grad = gradient(x, y)
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        trajectory.append((x, y))
    return np.array(trajectory)


def gradient_descent_scenario():
    learning_rate = 0.1
    iterations = 1000
    changes_of_input = gradient_descent_algorithm(iterations, learning_rate)
    # Extract x, y, and iteration values for plotting
    x_values = changes_of_input[:, 0]
    y_values = changes_of_input[:, 1]
    iterations = np.arange(iterations)
    # Print the final point
    final_x, final_y = x_values[-1], y_values[-1]
    print(f"Final Point: ({final_x}, {final_y})")
    # Plot the trajectory with colored points based on iteration number
    plt.scatter(x_values, y_values, c=iterations, cmap='viridis', marker='o', edgecolors='black')
    # Add labels and a colorbar
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Gradient Descent through iterations')
    cbar = plt.colorbar()
    cbar.set_label('Iterations')
    # Show the plot
    plt.show()


def logistic_regression_scenario():
    x_train, y_train = read_data_demo('train.csv')
    x_validation, y_validation = read_data_demo('validation.csv')
    x_test, y_test = read_data_demo('test.csv')

    # Implement torches from NumPy arrays
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_validation = torch.from_numpy(x_validation)
    y_validation = torch.from_numpy(y_validation)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    validation_dataset = torch.utils.data.TensorDataset(x_validation, y_validation)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    # Create PyTorch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model features
    num_epochs = 10
    criterion = torch.nn.CrossEntropyLoss()

    for learning_rate in [0.1, 0.01, 0.001]:
        model = Logistic_Regression(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        epoch_train_loss_values, epoch_validation_loss_values, epoch_test_loss_values = [], [], []

        for epoch in range(num_epochs):
            # Training
            model.train()  # Set the model to training mode
            train_loss_values = []
            ep_train_correct_preds = 0.0
            # Iterate over batches in the training dataset
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients to prevent accumulation
                outputs = model.forward(inputs)  # Forward pass
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update the weights using the gradients
                # Store the loss values for plotting
                train_loss_values.append(loss.item())
                ep_train_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            # Calculate train loss values for entire epoch
            train_mean_loss = np.mean(train_loss_values)
            ep_train_accuracy = ep_train_correct_preds / len(train_dataset)

            # Validation
            model.eval()  # Set the model to evaluation mode
            validation_loss_values = []
            ep_validation_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in validation_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    validation_loss_values.append(loss.item())
                    ep_validation_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            validation_mean_loss = np.mean(validation_loss_values)
            ep_validation_accuracy = ep_validation_correct_preds / len(validation_dataset)

            # Test
            model.eval()  # Set the model to evaluation mode
            test_loss_values = []
            ep_test_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in test_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    test_loss_values.append(loss.item())
                    ep_test_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            test_mean_loss = np.mean(test_loss_values)
            ep_test_accuracy = ep_test_correct_preds / len(test_dataset)

            # Printing
            epoch_train_loss_values.append(train_mean_loss)
            epoch_validation_loss_values.append(validation_mean_loss)
            epoch_test_loss_values.append(test_mean_loss)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_mean_loss.item():.4f}, '
                  f'Train accuracy: {ep_train_accuracy:.4f}, '
                  f'Validation loss: {validation_mean_loss.item():.4f}, '
                  f'Validation accuracy: {ep_validation_accuracy:.4f}, '
                  f'Test loss: {test_mean_loss.item():.4f}, Test accuracy: {ep_test_accuracy:.4f}')
        plot_decision_boundaries(model, x_test.numpy(), y_test.numpy(),
                                 f'Logistic Regression with {learning_rate} learning rate')

        # Plot the loss values through epochs
        # Plot training, validation, and test loss progression
        plt.plot(epoch_train_loss_values, marker='o', linestyle='',label='Training Loss', color='blue')
        plt.plot(epoch_validation_loss_values, marker='o', linestyle='', label='Validation Loss', color='orange')
        plt.plot(epoch_test_loss_values, marker='o', linestyle='',label='Test Loss', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Value')
        plt.title('Loss Progression of best Logistic Regression Model (0.001 learning rate)')
        plt.legend()  # Add legend to the plot
        plt.show()


def multi_classes_scenario():
    x_train, y_train = read_data_demo('train_multiclass.csv')
    x_validation, y_validation = read_data_demo('validation_multiclass.csv')
    x_test, y_test = read_data_demo('test_multiclass.csv')

    # Implement torches from NumPy arrays
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_validation = torch.from_numpy(x_validation)
    y_validation = torch.from_numpy(y_validation)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    validation_dataset = torch.utils.data.TensorDataset(x_validation, y_validation)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    # Create PyTorch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model features
    n_classes = len(torch.unique(y_test))
    num_epochs = 30
    criterion = torch.nn.CrossEntropyLoss()

    for learning_rate in [0.01, 0.001, 0.0003]:
        model = Logistic_Regression(2, n_classes)
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
        epoch_train_loss_values, epoch_validation_loss_values, epoch_test_loss_values = [], [], []
        epoch_train_accuracy_values, epoch_validation_accuracy_values, epoch_test_accuracy_values = [], [], []

        for epoch in range(num_epochs):
            # Training
            model.train()  # Set the model to training mode
            train_loss_values = []
            ep_train_correct_preds = 0.0
            # Iterate over batches in the training dataset
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients to prevent accumulation
                outputs = model.forward(inputs)  # Forward pass
                # Compute the loss
                loss = criterion(outputs.squeeze(), labels)
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update the weights using the gradients
                # Store the loss values for plotting
                train_loss_values.append(loss.item())
                ep_train_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            # Calculate train loss values for entire epoch
            train_mean_loss = np.mean(train_loss_values)
            ep_train_accuracy = ep_train_correct_preds / len(train_dataset)

            # Validation
            model.eval()  # Set the model to evaluation mode
            validation_loss_values = []
            ep_validation_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in validation_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    validation_loss_values.append(loss.item())
                    ep_validation_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            validation_mean_loss = np.mean(validation_loss_values)
            ep_validation_accuracy = ep_validation_correct_preds / len(validation_dataset)

            # Test
            model.eval()  # Set the model to evaluation mode
            test_loss_values = []
            ep_test_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in test_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    test_loss_values.append(loss.item())
                    ep_test_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            test_mean_loss = np.mean(test_loss_values)
            ep_test_accuracy = ep_test_correct_preds / len(test_dataset)

            # Update the decay in rate
            lr_scheduler.step()

            # Printing
            epoch_train_loss_values.append(train_mean_loss)
            epoch_validation_loss_values.append(validation_mean_loss)
            epoch_test_loss_values.append(test_mean_loss)
            epoch_train_accuracy_values.append(ep_train_accuracy)
            epoch_validation_accuracy_values.append(ep_validation_accuracy)
            epoch_test_accuracy_values.append(ep_test_accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_mean_loss.item():.4f}, '
                  f'Train accuracy: {ep_train_accuracy:.4f}, '
                  f'Validation loss: {validation_mean_loss.item():.4f}, '
                  f'Validation accuracy: {ep_validation_accuracy:.4f}, '
                  f'Test loss: {test_mean_loss.item():.4f}, Test accuracy: {ep_test_accuracy:.4f}')
        # Plot the loss values through epochs
        # Plot training, validation, and test loss progression
        plt.plot(epoch_train_loss_values, marker='o', linestyle='', label='Training Loss',
                 color='blue')
        plt.plot(epoch_validation_loss_values, marker='o', linestyle='', label='Validation Loss',
                 color='orange')
        plt.plot(epoch_test_loss_values, marker='o', linestyle='', label='Test Loss', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend()  # Add legend to the plot
        plt.xticks([1, 5, 10, 15, 20, 25, 30])
        plt.show()
        # Plot the accuracy values through epochs
        # Plot training, validation, and test accuracy progression
        plt.plot(epoch_train_accuracy_values, marker='o', linestyle='',label='Training Accuracy', color='blue')
        plt.plot(epoch_validation_accuracy_values, marker='o', linestyle='',label='Validation Accuracy', color='orange')
        plt.plot(epoch_test_accuracy_values, marker='o', linestyle='',label='Test Accuracy', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over epochs')
        plt.legend()  # Add legend to the plot
        plt.xticks([1, 5, 10, 15, 20, 25, 30])
        plt.show()


def decision_tree_classifier():
    x_train, y_train = read_data_demo('train_multiclass.csv')
    x_validation, y_validation = read_data_demo('validation_multiclass.csv')
    x_test, y_test = read_data_demo('test_multiclass.csv')
    decision_tree = DecisionTreeClassifier(max_depth=10)
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Decision Tree Accuracy:", accuracy)
    plot_decision_boundaries(decision_tree, x_test, y_test, "Decision Boundaries for Decision Tree (max_depth=10)")


def plot_accuracies_of_models():
    # Learning rates
    learning_rates = [0.01, 0.001, 0.0003]
    # Validation accuracies
    validation_accuracies = [0.8421, 0.7985, 0.8065]
    # Test accuracies
    test_accuracies = [0.8424, 0.7974, 0.8073]
    # Plot validation accuracies
    plt.scatter(learning_rates, validation_accuracies, color='yellow', label='Validation Accuracy')
    # Plot test accuracies
    plt.scatter(learning_rates, test_accuracies, color='blue', label='Test Accuracy')
    # Add labels and title
    plt.xlabel('Learning Rates')
    plt.ylabel('Accuracy')
    plt.title('Validation and Test Accuracies at Different Learning Rates')
    # Add legend
    plt.legend()
    # Show plot
    plt.show()


def multi_class_with_regularization_scenario():
    x_train, y_train = read_data_demo('train_multiclass.csv')
    x_validation, y_validation = read_data_demo('validation_multiclass.csv')
    x_test, y_test = read_data_demo('test_multiclass.csv')

    # Implement torches from NumPy arrays
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    y_test = torch.from_numpy(y_test)
    x_validation = torch.from_numpy(x_validation)
    y_validation = torch.from_numpy(y_validation)

    # Create PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    validation_dataset = torch.utils.data.TensorDataset(x_validation, y_validation)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    # Create PyTorch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=32,
                                                    shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model features
    n_classes = len(torch.unique(y_test))
    num_epochs = 30
    criterion = torch.nn.CrossEntropyLoss()

    for regularization_parameter in [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        model = Logistic_Regression(2, n_classes, regularization_parameter)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
        epoch_train_loss_values, epoch_validation_loss_values, epoch_test_loss_values = [], [], []
        epoch_train_accuracy_values, epoch_validation_accuracy_values, epoch_test_accuracy_values = [], [], []

        for epoch in range(num_epochs):
            # Training
            model.train()  # Set the model to training mode
            train_loss_values = []
            ep_train_correct_preds = 0.0
            # Iterate over batches in the training dataset
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Zero the gradients to prevent accumulation
                outputs = model.forward(inputs)  # Forward pass
                # Compute the loss
                loss = criterion(outputs.squeeze(), labels) + model.l2_regularization_loss()
                loss.backward()  # Backward pass to compute gradients
                optimizer.step()  # Update the weights using the gradients
                # Store the loss values for plotting
                train_loss_values.append(loss.item())
                ep_train_correct_preds += torch.sum(torch.argmax(outputs, dim=1) == labels).item()
            # Calculate train loss values for entire epoch
            train_mean_loss = np.mean(train_loss_values)
            ep_train_accuracy = ep_train_correct_preds / len(train_dataset)

            # Validation
            model.eval()  # Set the model to evaluation mode
            validation_loss_values = []
            ep_validation_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in validation_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    validation_loss_values.append(loss.item())
                    ep_validation_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            validation_mean_loss = np.mean(validation_loss_values)
            ep_validation_accuracy = ep_validation_correct_preds / len(validation_dataset)

            # Test
            model.eval()  # Set the model to evaluation mode
            test_loss_values = []
            ep_test_correct_preds = 0.0
            # Iterate over batches in the validation dataset
            with torch.no_grad():  # No need to compute gradients during validation
                for inputs, labels in test_loader:
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    # Store the loss values
                    test_loss_values.append(loss.item())
                    ep_test_correct_preds += torch.sum(
                        torch.argmax(outputs, dim=1) == labels).item()
            # Calculate validation loss values for entire epoch
            test_mean_loss = np.mean(test_loss_values)
            ep_test_accuracy = ep_test_correct_preds / len(test_dataset)

            # Update the decay in rate
            lr_scheduler.step()

            # Printing
            epoch_train_loss_values.append(train_mean_loss)
            epoch_validation_loss_values.append(validation_mean_loss)
            epoch_test_loss_values.append(test_mean_loss)
            epoch_train_accuracy_values.append(ep_train_accuracy)
            epoch_validation_accuracy_values.append(ep_validation_accuracy)
            epoch_test_accuracy_values.append(ep_test_accuracy)
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train loss: {train_mean_loss.item():.4f}, '
                  f'Train accuracy: {ep_train_accuracy:.4f}, '
                  f'Validation loss: {validation_mean_loss.item():.4f}, '
                  f'Validation accuracy: {ep_validation_accuracy:.4f}, '
                  f'Test loss: {test_mean_loss.item():.4f}, Test accuracy: {ep_test_accuracy:.4f}')
        plot_decision_boundaries(model, x_test.numpy(), y_test.numpy(),
                                 f'Logistic Regression with regularization parameter {regularization_parameter}')
        # Plot the loss values through epochs
        # Plot training, validation, and test loss progression
        plt.plot(epoch_train_loss_values, marker='o', linestyle='', label='Training Loss',
                 color='blue')
        plt.plot(epoch_validation_loss_values, marker='o', linestyle='', label='Validation Loss',
                 color='orange')
        plt.plot(epoch_test_loss_values, marker='o', linestyle='', label='Test Loss', color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss over epochs')
        plt.legend()  # Add legend to the plot
        plt.xticks([1, 5, 10, 15, 20, 25, 30])
        plt.show()
        # Plot the accuracy values through epochs
        # Plot training, validation, and test accuracy progression
        plt.plot(epoch_train_accuracy_values, marker='o', linestyle='', label='Training Accuracy',
                 color='blue')
        plt.plot(epoch_validation_accuracy_values, marker='o', linestyle='',
                 label='Validation Accuracy', color='orange')
        plt.plot(epoch_test_accuracy_values, marker='o', linestyle='', label='Test Accuracy',
                 color='green')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over epochs')
        plt.legend()  # Add legend to the plot
        plt.xticks([1, 5, 10, 15, 20, 25, 30])
        plt.show()

if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    # ridge_regression_scenario()
    # gradient_descent_scenario()
    # logistic_regression_scenario()
    # multi_classes_scenario()
    # decision_tree_classifier()
    # plot_accuracies_of_models()
    # multi_class_with_regularization_scenario()