class FeedForwardNetwork:
    def __init__(self, layers):
        """Initialize the feedforward network with a list of layers."""
        self.layers = layers

    def forward(self, inputs):
        """Perform a forward pass through the network."""
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, training_data, epochs, learning_rate):
        """Train the network using the provided training data."""
        for epoch in range(epochs):
            for inputs, target in training_data:
                # Forward pass and backward pass would go here
                pass
