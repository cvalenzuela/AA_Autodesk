/*
A very simple introduction to tensorflow.js
Based on https://goo.gl/YmuuhK
Cristobal Valenzuela
cris@runwayml.com
License MIT
*/

// Lets start by defining some synthetic data
// The relationship between x and y is y = 2x - 1
// We will train a model to predict a linear relationship
const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

// We start by creating a function that will train a model
// This function is asynchronous, because it will take some time to run
async function learnLinear() {
  // Here we create a new model.
  // In a sequential model de outputs of one model
  // are the inputs of the next
  // No branching or skip layers
  const model = tf.sequential();

  // We add a Dense layer.
  // Here, all of the nodes in each of the layers 
  // are connected to each other
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  // We then need to compile our model
  // We define meanSquaredError as our loss function
  // and stochastic gradient descent as the optimizer
  model.compile({
    loss: 'meanSquaredError',
    optimizer: 'sgd'
  });

  // We then need to train our model by
  // calling model.fit
  await model.fit(xs, ys, { epochs: 250 });

  // We can then log our result to the page
  const prediction = model.predict(tf.tensor2d([20], [1, 1]));
  document.getElementById('output_field').innerText = prediction;
}

// Here we start the process of training
learnLinear();