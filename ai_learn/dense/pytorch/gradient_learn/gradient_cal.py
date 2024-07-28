import torch
import torch.optim as optim

def gradient_with_one_variable():
    # Define the tensor and enable gradient computation
    x = torch.tensor(2.0, requires_grad=True)

    # Define the function
    y = x**2 + 2*x + 1

    # Perform backpropagation to compute the gradient
    y.backward()

    # print("before update weight", x)
    # optimizer = optim.SGD([x], lr=0.1)
    # optimizer.step()
    # print("after update weight", x)

    # Output the gradient
    print(f"The gradient of y with respect to x at x=2.0 is: {x.grad}")

def gradient_with_two_variables():

    # Define tensors for x and y with gradient tracking
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    # Define the function
    z = 3 * x ** 2 + 4 * y ** 2 + x * y

    # Perform backpropagation
    z.backward()

    # Output the gradients
    print(f"Gradient of z with respect to x at (x=2.0, y=3.0) is: {x.grad}")
    print(f"Gradient of z with respect to y at (x=2.0, y=3.0) is: {y.grad}")


def simple_linear_regression():

    # Define the dataset (input and output)
    x_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y_data = torch.tensor([2.0, 4.0, 6.0, 8.0])

    # Initialize weights and bias with gradient tracking
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    # Define the learning rate
    learning_rate = 0.01

    # Define the optimizer
    optimizer = optim.SGD([w, b], lr=learning_rate)

    # Training loop
    for epoch in range(100):
        # Compute predictions
        y_pred = w * x_data + b

        # Compute the loss (Mean Squared Error)
        loss = ((y_pred - y_data) ** 2).mean()

        # Perform backpropagation to compute gradients
        loss.backward()

        # Update the weights using the optimizer
        optimizer.step()

        # Zero the gradients
        optimizer.zero_grad()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: w={w.item():.3f}, b={b.item():.3f}, loss={loss.item():.3f}")

    # Final model parameters
    print(f"Final: w={w.item():.3f}, b={b.item():.3f}")


if __name__ == '__main__':
    gradient_with_one_variable()
    # gradient_with_two_variables()
    # simple_linear_regression()