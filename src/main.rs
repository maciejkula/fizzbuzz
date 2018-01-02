extern crate rand;
extern crate wyrm;

use std::ops::Deref;

use wyrm::*;

fn layer_init(rows: usize, cols: usize) -> Arr {
    Arr::zeros((rows, cols)).map(|_| rand::random::<f32>() / (cols as f32).sqrt())
}

fn to_binary(number: u64, arr: &mut Arr) {
    for i in 0..64 {
        arr[(0, i)] = match number & (1 << i) {
            0 => 0.0,
            _ => 1.0,
        }
    }
}

fn to_label(number: u64) -> usize {
    if number % 3 == 0 && number % 5 == 0 {
        3
    } else if number % 5 == 0 {
        2
    } else if number % 3 == 0 {
        1
    } else {
        0
    }
}

fn predicted_label(softmax_output: &Arr) -> usize {
    softmax_output
        .iter()
        .enumerate()
        .max_by(|&(_, x), &(_, y)| x.partial_cmp(y).unwrap())
        .unwrap()
        .0
}

fn main() {
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

    // Optimizer
    let mut optimizer = SGD::new(0.2, vec![dense_1.clone(), dense_2.clone(), dense_3.clone()]);

    let mut input_array = Arr::zeros((1, number_size));
    let mut labels = Arr::zeros((1, num_classes));

    let num_epochs = 1000;

    for epoch in 0..num_epochs {
        let mut loss_value = 0.0;
        let mut correct = 0;
        let mut total = 0;

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

            // Reset the graph
            loss.zero_gradient();

            loss_value += loss.value().scalar_sum();

            // Get the prediction
            let softmax_output = prediction.value();
            if predicted_label(softmax_output.deref()) == label_idx {
                correct += 1;
            }

            total += 1;
        }

        if epoch % 10 == 0 {
            println!(
                "Epoch {}: loss {}, accuracy {}",
                epoch,
                loss_value,
                correct as f32 / total as f32
            );

            for &number in [1, 3, 5, 8, 11, 15, 300].iter() {
                prediction.zero_gradient();
                to_binary(number as u64, &mut input_array);
                binary_input.set_value(&input_array);
                prediction.forward();
                let label = predicted_label(prediction.value().deref());

                let predicted_string = match label {
                    0 => format!("{}", number),
                    1 => "Fizz".into(),
                    2 => "Buzz".into(),
                    3 => "FizzBuzz".into(),
                    _ => panic!("This shouldn't happen."),
                };

                println!("    {} -> {}", number, predicted_string);
            }
        }
    }
}
