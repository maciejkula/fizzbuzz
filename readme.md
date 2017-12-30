# FizzBuzz

A simple solution to the FizzBuzz problem. Using a three-layer perceptron implemented using [wyrm](https://github.com/maciejkula/wyrm), my autodiff library.

To run:

1. [Install Rust](https://www.rustup.rs/). You will need the nightly compiler (`rustup install nightly && rustup default nightly`).
2. Clone this repo: `git clone git@github.com:maciejkula/fizzbuzz.git`.
3. Run via `cd fizzbuzz && cargo run --release`.

You should see something like the following:
```
   Compiling fizzbuzz v0.1.0 (file:///home/maciej/Code/fizzbuzz)
    Finished release [optimized] target(s) in 9.1 secs
     Running `target/release/fizzbuzz`
Epoch 0: loss 1368.3994, accuracy 0.5095095
    1 -> 
    3 -> 
    5 -> 
    8 -> 
    11 -> 
    15 -> 
    300 -> 

<snip>

Epoch 980: loss 0.250431, accuracy 1
    1 -> 
    3 -> Fizz
    5 -> Buzz
    8 -> 
    11 -> 
    15 -> FizzBuzz
    300 -> FizzBuzz

```

## Model definition
The model is a simple 3-layer perceptron working off binary representations of unsigned integers:
```rust
let number_size = 64;
let num_classes = 4; // Nothing, Fizz, Buzz, and FizzBuzz
let latent_dim = 64;

let binary_input = InputNode::new(Arr::zeros((1, number_size)));
let output = InputNode::new(Arr::zeros((1, num_classes)));

let dense_1 = ParameterNode::new(layer_init(number_size, latent_dim));
let dense_2 = ParameterNode::new(layer_init(latent_dim, latent_dim));
let dense_3 = ParameterNode::new(layer_init(latent_dim, num_classes));

let prediction = binary_input
    .clone()
    .dot(&dense_1)
    .sigmoid()
    .dot(&dense_2)
    .sigmoid()
    .dot(&dense_3)
    .softmax();

// Cross-entropy
let mut loss = (-(output.clone() * prediction.ln())).scalar_sum();
```

The training loop generates new training examples and takes SGD steps:
```
for number in 1..1000 {
    // Binary-encode the input
    to_binary(number, &mut input_array);

    // Set the output
    let label_idx = to_label(number);

    labels.fill(0.0);
    labels[(0, label_idx)] = 1.0;
    
    // Pass inputs to graph
    binary_input.set_value(&input_array);
    output.set_value(&labels);
    
    // Run forward pass
    loss.forward();
    
    // Run backward pass with weight 1
    loss.backward(1.0);
    
    // Update the parameters
    optimizer.step();
    optimizer.zero_gradients();
```