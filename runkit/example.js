/* eslint-disable no-console */

// Load training data from Iris Data Set obtained
// from UCI Machine Learning Repository.
const trainingData = require( 'wink-perceptron/sample-data/iris-train.json' );
// Initialize a test data sample.
const testData = {
  setosa: { sepalLength: 4.9, sepalWidth: 3, petalLength: 1.4, petalWidth: 0.2 },
  versicolor: { sepalLength: 6.4, sepalWidth: 3.2, petalLength: 4.5, petalWidth: 1.5 },
  virginica: { sepalLength: 7.2, sepalWidth: 3.6, petalLength: 6.1, petalWidth: 2.5 }
};

// Load wink perceptron.
var winkPerceptron = require( 'wink-perceptron' );
// Instantiate wink perceptron.
var perceptron = winkPerceptron();
// Define configurtaion.
perceptron.defineConfig( { shuffleData: true, maxIterations: 21 } );
// Learn from training data.
perceptron.learn( trainingData );

// Attempt prediction for each iris plant type.
console.log( perceptron.predict( testData.setosa ) );
// -> Iris-setosa
console.log( perceptron.predict( testData.versicolor ) );
// -> Iris-versicolor
console.log( perceptron.predict( testData.virginica ) );
// -> Iris-virginica
