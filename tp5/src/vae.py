import numpy as np
from .perceptron import MultiLayerPerceptron
from .utils import Utils


class VAE(MultiLayerPerceptron):
    def __init__(self, encoderArchitecture, optimizerClass, optimizerOptions, inputs):
        self.latentSpaceIndex = len(encoderArchitecture) - 1
        self.latentSpaceDimension = encoderArchitecture[-1]["neuronQty"]
        architecture = Utils.generateAutoencoderArchitecture(encoderArchitecture, len(inputs[0]))
        self.inputs = inputs
        super().__init__(architecture, optimizerClass, optimizerOptions)

    def train(self):
        expectedOutputs = []
        for input in self.inputs:
            expectedOutputs.append(input[1:])
        
        for epoch in range(self.constants.maxEpochs):
            total_loss = 0
            for input, expectedOutput in zip(self.inputs, expectedOutputs):
                # Forward pass
                outputs = []
                for i, layer in enumerate(self.layers):
                    layerInput = outputs[i-1] if i > 0 else input
                    layerOutput = np.array([n.test(layerInput) for n in layer])
                    if i == self.latentSpaceIndex: # I'm on the latent space
                        mu = layerOutput[0]
                        sigma = layerOutput[1]
                        print(f"mu: {mu}, sigma: {sigma}")
                        z = Utils.sample(mu, sigma) # z sample calculation
                        layerOutput = np.insert(z, 0, self.constants.bias)
                    elif i != len(self.layers) - 1:
                        layerOutput = np.insert(layerOutput, 0, self.constants.bias)
                    outputs.append(layerOutput)
                
                # Calculate reconstruction loss
                reconstruction_loss = np.sum((expectedOutput - outputs[-1][1:]) ** 2)
                
                # Calculate KL divergence
                kl_divergence = -0.5 * np.sum(1 + np.log(sigma ** 2) - mu ** 2 - sigma ** 2)
                
                # Total loss
                loss = reconstruction_loss + kl_divergence
                total_loss += loss
                
                # Backpropagation
                deltas = []
                for i, layer in reversed(list(enumerate(self.layers))):
                    layerDeltas = []
                    if i == len(self.layers) - 1:
                        for k, n in enumerate(layer):
                            delta = (expectedOutput[k] - outputs[i][k+1]) * n.activationFunction.derivative(outputs[i-1], n.weights)
                            layerDeltas.append(delta)
                            deltaW = n.optimizer(delta * outputs[i-1], previousDeltaW=n.weights - n.weightsHistory[-2] if len(n.weightsHistory) > 1 else 0)
                            n.incrementDeltaW(deltaW)
                    elif i == self.latentSpaceIndex:
                        for k, n in enumerate(layer):
                            delta = (deltas[-1][k] * sigma[k] + mu[k]) * n.activationFunction.derivative(outputs[i-1], n.weights)
                            layerDeltas.append(delta)
                            deltaW = n.optimizer(delta * outputs[i-1], previousDeltaW=n.weights - n.weightsHistory[-2] if len(n.weightsHistory) > 1 else 0)
                            n.incrementDeltaW(deltaW)
                    else:
                        for k, n in enumerate(layer):
                            weightsBetweenMeAndNextLayer = np.array([r.weights[k+1] for r in self.layers[i+1]])
                            delta = np.dot(deltas[-1], weightsBetweenMeAndNextLayer) * n.activationFunction.derivative(outputs[i-1], n.weights)
                            layerDeltas.append(delta)
                            deltaW = n.optimizer(delta * outputs[i-1], previousDeltaW=n.weights - n.weightsHistory[-2] if len(n.weightsHistory) > 1 else 0)
                            n.incrementDeltaW(deltaW)
                    deltas.append(layerDeltas)
                
                for layer in self.layers:
                    for n in layer:
                        n.updateWeights()
            
            # Print the total loss for the epoch
            print(f"Epoch {epoch + 1}/{self.constants.maxEpochs}, Loss: {total_loss}")
            
            if np.abs(total_loss) <= self.constants.epsilon:
                for layer in self.layers:
                    for neuron in layer:
                        neuron.saveWeightsForEpoch()
                return
            for layer in self.layers:
                for neuron in layer:
                    neuron.saveWeightsForEpoch()
        print("Training finished without convergence.")

    def getLatentSpaceOutput(self, input):
        outputs = []
        for i in range(self.latentSpaceIndex + 1):
            layer = self.layers[i]
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]

    def testWithLatentSpaceInput(self, input):
        if len(input) != self.latentSpaceDimension:
            raise ValueError("Input must have the same dimension as the latent space")

        input = np.insert(input, 0, self.constants.bias)
        outputs = []
        for i in range(self.latentSpaceIndex, len(self.layers)):
            layer = self.layers[i]
            layerInput = outputs[i-1] if i>0 else input
            layerOutput = np.array(list(map(lambda n: n.test(layerInput), layer)))
            if i != len(self.layers)-1:
                layerOutput = np.insert(layerOutput, 0, self.constants.bias)
            outputs.append(layerOutput)
        return outputs[-1]

    def generateFromLatentSpace(self, num_samples):
        samples = []
        for _ in range(num_samples):
            z = np.random.normal(0, 1, self.latentSpaceDimension) 
            z = np.insert(z, 0, self.constants.bias) 
            outputs = []
            for i in range(self.latentSpaceIndex, len(self.layers)):
                layer = self.layers[i]
                layerInput = outputs[i-1] if i > self.latentSpaceIndex else z
                layerOutput = np.array([n.test(layerInput) for n in layer])
                if i != len(self.layers) - 1:
                    layerOutput = np.insert(layerOutput, 0, self.constants.bias)
                outputs.append(layerOutput)
            samples.append(outputs[-1])
        return samples
