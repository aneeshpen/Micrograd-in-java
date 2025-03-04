import java.util.*;
import java.util.Arrays;

public class TestMLP {

    public static void main(String[] args) {
        // Create N training examples with 2 features each.
        // Label: 1.0 if (x1 + x2 > 1.0), otherwise -1.0.
        int N = 100;  // number of examples
        List<List<Value>> X = new ArrayList<>();
        List<Double> Y = new ArrayList<>();
        Random rand = new Random(42); // fixed seed for reproducibility

        for (int i = 0; i < N; i++) {
            double x1 = rand.nextDouble(); // in [0, 1)
            double x2 = rand.nextDouble();
            double target = (x1 + x2 > 1.0) ? 1.0 : -1.0;
            // Each input is represented as a list of Value objects.
            X.add(Arrays.asList(new Value(x1), new Value(x2)));
            Y.add(target);
        }

        // MLP with 2 input neurons, one hidden layer with 4 neurons,
        // and an output layer with 1 neuron.
        // Here, the hidden layer neurons use ReLU (by default), and the output layer remains linear.
        // We will apply tanh on the output manually.
        MLP mlp = new MLP(2, new int[]{4, 1});
        double learningRate = 0.1;
        int epochs = 100;

        // Training
        for (int epoch = 0; epoch < epochs; epoch++) {
            // Zero gradients from previous iteration.
            mlp.zeroGrad();

            // Initialize total loss.
            Value totalLoss = new Value(0.0);

            // Process each training example.
            for (int i = 0; i < N; i++) {
                List<Value> x = X.get(i);
                double target = Y.get(i);
                // Forward pass
                List<Value> out = mlp.forward(x);
                // Apply tanh activation at the output.
                Value pred = out.get(0).tanh();
                // Compute mean squared error (MSE) loss
                Value diff = pred.sub(new Value(target));
                Value loss = diff.pow(2);
                totalLoss = totalLoss.add(loss);
            }
            
            totalLoss = totalLoss.div(N);

            // Backward pass
            totalLoss.backward();

            // Gradient descent update
            for (Value p : mlp.parameters()) {
                p.data -= learningRate * p.grad;
            }

            // Print training loss every 10 epochs.
            if (epoch % 10 == 0) {
                System.out.println("Epoch " + epoch + " Loss: " + totalLoss.data);
            }
        }

        
        System.out.println("\nFinal predictions on training data:");
        for (int i = 0; i < N; i++) {
            List<Value> x = X.get(i);
            List<Value> out = mlp.forward(x);
            Value pred = out.get(0).tanh();
            int prediction = (pred.data > 0) ? 1 : -1;
            System.out.println("Input: (" + x.get(0).data + ", " + x.get(1).data + ") " +
                               " Prediction: " + prediction + " (tanh output: " + pred.data + ")");
        }
    }
}
