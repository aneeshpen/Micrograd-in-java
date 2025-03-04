import java.util.*;
import java.lang.Math;

public class Value {
    public double data;
    public double grad = 0.0;
    public Set<Value> _prev;
    public String _op;
    public String label;
    public BackwardFunction _backward;

    public Value(double data) {
        this.data = data;
        this._prev = new HashSet<>();
        this._op = "";
        this.label = "";
        this._backward = () -> {};
    }

    public Value(double data, Set<Value> children, String op, String label) {
        this.data = data;
        this._prev = children;
        this._op = op;
        this.label = label;
        this._backward = () -> {};
    }

    @Override
    public String toString() {
        return "Value(data=" + data + ")";
    }

    // Addition
    public Value add(Value other) {
        Value output = new Value(this.data + other.data,
                new HashSet<>(Arrays.asList(this, other)), "+", "");
        output._backward = () -> {
            this.grad += 1.0 * output.grad;
            other.grad += 1.0 * output.grad;
        };
        return output;
    }

    public Value add(double other) {
        return this.add(new Value(other));
    }

    // Negation
    public Value negate() {
        return this.mul(-1);
    }

    // Subtraction
    public Value sub(Value other) {
        return this.add(other.negate());
    }

    public Value sub(double other) {
        return this.sub(new Value(other));
    }

    // Multiplication
    public Value mul(Value other) {
        Value output = new Value(this.data * other.data,
                new HashSet<>(Arrays.asList(this, other)), "*", "");
        output._backward = () -> {
            this.grad += other.data * output.grad;
            other.grad += this.data * output.grad;
        };
        return output;
    }

    public Value mul(double other) {
        return this.mul(new Value(other));
    }

    // Division: defined as multiplication by the reciprocal.
    public Value div(Value other) {
        return this.mul(other.pow(-1));
    }

    public Value div(double other) {
        return this.div(new Value(other));
    }

    // Exponent
    public Value pow(double exponent) {
        Value output = new Value(Math.pow(this.data, exponent),
                new HashSet<>(Arrays.asList(this)), "**" + exponent, "");
        output._backward = () -> {
            this.grad += exponent * Math.pow(this.data, exponent - 1) * output.grad;
        };
        return output;
    }

    // tanh activation function
    public Value tanh() {
        double t = Math.tanh(this.data);
        Value output = new Value(t,
                new HashSet<>(Arrays.asList(this)), "tanh", "");
        output._backward = () -> {
            this.grad += (1 - t * t) * output.grad;
        };
        return output;
    }

    // Exponential. e power x 
    public Value exp() {
        double expValue = Math.exp(this.data);
        Value output = new Value(expValue,
                new HashSet<>(Arrays.asList(this)), "exp", "");
        output._backward = () -> {
            this.grad += output.data * output.grad;
        };
        return output;
    }

    // ReLU activation function
    public Value relu() {
        double outData = this.data < 0 ? 0 : this.data;
        Value output = new Value(outData,
                new HashSet<>(Arrays.asList(this)), "ReLU", "");
        output._backward = () -> {
            this.grad += (this.data > 0 ? 1.0 : 0.0) * output.grad;
        };
        return output;
    }

    // Backward pass using topologiccal sort
    public void backward() {
        List<Value> topo = new ArrayList<>();
        Set<Value> visited = new HashSet<>();
        buildTopo(this, visited, topo);
        this.grad = 1.0;
        Collections.reverse(topo);
        for (Value node : topo) {
            node._backward.apply();
        }
    }

    private void buildTopo(Value v, Set<Value> visited, List<Value> topo) {
        if (!visited.contains(v)) {
            visited.add(v);
            for (Value child : v._prev) {
                buildTopo(child, visited, topo);
            }
            topo.add(v);
        }
    }
}


@FunctionalInterface
interface BackwardFunction {
    void apply();
}

