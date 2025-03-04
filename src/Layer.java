import java.util.*;

// A layer consisting of multiple neurons.

public class Layer extends Module {
    public List<Neuron> neurons;

    /**
     * Constructs a layer.
     * @param nin Number of input features.
     * @param nout Number of neurons (outputs) in this layer.
     * @param nonlin Whether each neuron should apply a nonlinearity.
     */
    public Layer(int nin, int nout, boolean nonlin) {
        neurons = new ArrayList<>();
        for (int i = 0; i < nout; i++) {
            neurons.add(new Neuron(nin, nonlin));
        }
    }

    /**
     * Applies the layer to the input.
     * @param x List of input Values.
     * @return A list of output Values produced by each neuron.
     */
    public List<Value> forward(List<Value> x) {
        List<Value> outs = new ArrayList<>();
        for (Neuron n : neurons) {
            outs.add(n.forward(x));
        }
        return outs;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Neuron n : neurons) {
            params.addAll(n.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Layer of [");
        for (int i = 0; i < neurons.size(); i++) {
            sb.append(neurons.get(i).toString());
            if (i < neurons.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}

