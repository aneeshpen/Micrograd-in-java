import java.util.*;

public class MLP extends Module {
    public List<Layer> layers;

    /**
     * Constructs an MLP.
     * @param nin Number of input features.
     * @param nouts Array representing the number of neurons in each layer.
     *              For example, new int[]{16, 16, 1} creates a 3-layer network.
     */
    public MLP(int nin, int[] nouts) {
        layers = new ArrayList<>();
        int szPrev = nin;
        for (int i = 0; i < nouts.length; i++) {
            // Use nonlinearity for all but the last layer.
            boolean nonlin = (i != nouts.length - 1);
            layers.add(new Layer(szPrev, nouts[i], nonlin));
            szPrev = nouts[i];
        }
    }

    /**
     * Applies the MLP to the input.
     * @param x List of input Values.
     * @return The output of the network as a list of Values.
     */
    public List<Value> forward(List<Value> x) {
        List<Value> out = x;
        for (Layer layer : layers) {
            out = layer.forward(out);
        }
        return out;
    }

    @Override
    public List<Value> parameters() {
        List<Value> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("MLP of [");
        for (int i = 0; i < layers.size(); i++) {
            sb.append(layers.get(i).toString());
            if (i < layers.size() - 1) {
                sb.append(", ");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}

