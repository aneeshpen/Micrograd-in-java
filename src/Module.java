import java.util.*;

public abstract class Module {
    
    //Resets gradients of all parameters to zero.
     
    public void zeroGrad() {
        for (Value p : parameters()) {
            p.grad = 0;
        }
    }

    //Returns a list of parameters for this module.

    public List<Value> parameters() {
        return new ArrayList<>();
    }
}

