# wink-perceptron

Multi-class averaged perceptron

### [![Build Status](https://api.travis-ci.org/winkjs/wink-perceptron.svg?branch=master)](https://travis-ci.org/winkjs/wink-perceptron) [![Coverage Status](https://coveralls.io/repos/github/winkjs/wink-perceptron/badge.svg?branch=master)](https://coveralls.io/github/winkjs/wink-perceptron?branch=master) [![Inline docs](http://inch-ci.org/github/winkjs/wink-perceptron.svg?branch=master)](http://inch-ci.org/github/winkjs/wink-perceptron) [![dependencies Status](https://david-dm.org/winkjs/wink-perceptron/status.svg)](https://david-dm.org/winkjs/wink-perceptron) [![devDependencies Status](https://david-dm.org/winkjs/wink-perceptron/dev-status.svg)](https://david-dm.org/winkjs/wink-perceptron?type=dev) [![Gitter](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/winkjs/Lobby)

[<img align="right" src="https://decisively.github.io/wink-logos/logo-title.png" width="100px" >](http://winkjs.org/)

Wink Perceptron is a fast and effective way to learn linearly separable patterns from either dense or sparse data. Its averaging function results in better generalization compared to the vanilla implementation of perceptron.

### Installation

Use [npm](https://www.npmjs.com/package/wink-perceptron) to install:

    npm install wink-perceptron --save

### Getting Started
Here is an example of predicting type of iris plant from the [Iris Data Set](https://archive.ics.uci.edu/ml/datasets/iris).

```javascript
// Load training data — the Iris Data Set obtained from
// UCI Machine Learning Repository; it has been converted
// into JSON format.
// You may need to update the path in the "require" statement
// according to your working directory.
const trainingExamples = require( 'wink-perceptron/sample-data/iris-train.json' );
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
perceptron.learn( trainingExamples );

// Attempt prediction for each iris plant type.
console.log( perceptron.predict( testData.setosa ) );
// -> Iris-setosa
console.log( perceptron.predict( testData.versicolor ) );
// -> Iris-versicolor
console.log( perceptron.predict( testData.virginica ) );
// -> Iris-virginica
```
Try [experimenting with this example on Runkit](https://npm.runkit.com/wink-perceptron) in the browser.

### Documentation
Check out the [perceptron API documentation](http://winkjs.org/wink-perceptron/) to learn more.

### Need Help?

If you spot a bug and the same has not yet been reported, raise a new [issue](https://github.com/winkjs/wink-perceptron/issues) or consider fixing it and sending a pull request.

### About wink
[Wink](http://winkjs.org/) is a growing family of high quality [packages](http://winkjs.org/packages.html) for **Statistical Analysis**, **Natural Language Processing** and **Machine Learning** in NodeJS. The code is throughly documented for easy comprehension and has a test coverage of ~100% for reliability.

### Copyright & License

**wink-perceptron** is copyright 2017-18 [GRAYPE Systems Private Limited](http://graype.in/).

It is licensed under the under the terms of the GNU Affero General Public License as published by the Free Software Foundation, version 3 of the License.
