import java.util.*;

//A single neuron that computes a weighted sum of inputs with an optional nonlinearity.
public class Neuron extends Module {
    public List<Value> w;  // Weights
    public Value b;        // Bias
    public boolean nonlin; // nonlinearity

    /**
     * Constructs a neuron with the given number of inputs and an optional nonlinearity.
     * @param nin Number of input features.
     * @param nonlin If true, applies ReLU activation.
     */
    public Neuron(int nin, boolean nonlin) {
        this.nonlin = nonlin;
        this.w = new ArrayList<>();
        for (int i = 0; i < nin; i++) {
            // Random weight between -1 and 1.
            double randVal = Math.random() * 2 - 1;
            this.w.add(new Value(randVal));
        }
        this.b = new Value(0.0);
    }

    /**
     * Constructs a neuron with nonlinearity enabled by default.
     * @param nin Number of input features.
     */
    public Neuron(int nin) {
        this(nin, true);
    }

    /**
     * Computes the neuron's output given a list of input Values.
     * @param x List of input Values.
     * @return The computed output Value.
     */
    public Value forward(List<Value> x) {
        if (x.size() != w.size()) {
            throw new IllegalArgumentException("Input size does not match number of weights.");
        }
        // act = sum(w[i] * x[i]) + b
        Value act = this.b;
        for (int i = 0; i < x.size(); i++) {
            act = act.add(w.get(i).mul(x.get(i)));
        }
        return nonlin ? act.relu() : act;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>(w);
        params.add(b);
        return params;
    }

    @Override
    public String toString() {
        return (nonlin ? "ReLU" : "Linear") + "Neuron(" + w.size() + ")";
    }
}

