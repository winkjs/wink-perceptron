//     wink-perceptron
//     Multi-class averaged perceptron
//
//     Copyright (C) 2017-18  GRAYPE Systems Private Limited
//
//     This file is part of “wink-perceptron”.
//
//     Permission is hereby granted, free of charge, to any person obtaining a
//     copy of this software and associated documentation files (the "Software"),
//     to deal in the Software without restriction, including without limitation
//     the rights to use, copy, modify, merge, publish, distribute, sublicense,
//     and/or sell copies of the Software, and to permit persons to whom the
//     Software is furnished to do so, subject to the following conditions:
//
//     The above copyright notice and this permission notice shall be included
//     in all copies or substantial portions of the Software.
//
//     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//     THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//     FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//     DEALINGS IN THE SOFTWARE.

//
var helpers = require( 'wink-helpers' );
var shuffleArray = helpers.array.shuffle;
/* eslint-disable guard-for-in */

// wink multi-class averaged perceptron; its implementation is inspired by the
// research titled, "Practical Structured Learning Techniques for Natural Language
// Processing" by Harold Charles Daume dated August 2006.
//
// The essential idea is to (a) avoid addition of entire vector at every iteration,
// (b) perform computations only when updates occur, and (c) completely leverage
// the sparisity (if any) present in the vector. The  weight and bias adjustment
// funtions capture this idea.

// ### wink-perceptron
/**
 *
 * Creates an instance of {@link Perceptron}.
 *
 * @return {Perceptron} Object conatining set of API methods for preceptron
 * training, prediction, etc.
 * @example
 * // Load wink perceptron.
 * var perceptron = require( 'wink-perceptron' );
 * // Create your instance of wink perceptron.
 * var myPerceptron = perceptron();
*/
var perceptron = function () {
  // Learning Variables.
  // These are re-initialized in the `reset()` method.
  // The weights matrix with **features** x **classes** dimensions.
  var weights = Object.create( null );
  // Bias for every **class**.
  var biases = Object.create( null );
  // Following set of five variables are used to compute averages. The average
  // is computed by dividing the accumulated sums by updates.
  // Sum of weights used for comuting average weight.
  var sumOfWts = Object.create( null );
  // Alias for above: during call to `averageBalance()` sum is divided by updates
  // to turn it into average.
  var avgWts = sumOfWts;
  // Sum of biases used for computing average bias.
  var sumOfBiases = Object.create( null );
  // Alias for above: during call to `averageBalance()` sum is divided by updates
  // to turn it into average.
  var avgBiases = sumOfBiases;
  // Captures the last moment/epoch when an update of weight for a class/feature
  // combo has occurred. The moment/epoch here is nothing but the update number.
  var lastWtUpdatedAt = Object.create( null );
  // Captures the last iteration when an update of bias for a class occurred.
  var lastBsUpdatedAt = Object.create( null );
  // The number of updates.
  var updates = 0;
  // Number of examples seen.
  var examplesSeen = 0;
  // Imported flag to allow prediction without learning.
  var imported = false;

  // Configuration Variables and their default values.
  // Maximum number of learning iterations.
  var maxIterations = 9;
  // True means data will be shuffled after every iteration.
  var shuffleData = false;
  // Features Extractor function — used to extract features from each element of
  // the `data` that is passed to learn api. This ensures that shuffling occurs
  // at the `data` array level and not at feature level.
  var featureExtractor = null;

  /**
   * @classdesc Multi-class Averaged Perceptron  class.
   * @class Perceptron
   * @hideconstructor
   */
  var methods = Object.create( null );

  // ### predictUsingSpecificWeights
  /**
   *
   * Predicts the label for the input `features` using the `specificWeights` and
   * `specificBiases`. For example during learning process `weights` and `biases`
   * are used.
   *
   * If it is unable to predict then it returns a value **`unknown`**.
   *
   * @private
   * @param {object} features object that contains **name/value** pairs for every
   * feature.
   * @param {object} specificWeights these are either `weights` or `avgWts`.
   * @param {object} specificBiases these are either `biases` or `avgBiases`.
   *
   * @return {string} Predicted class label for the input `features`.
  */
  var predictUsingSpecificWeights = function ( features, specificWeights, specificBiases ) {
    // Scores, index by **class**.
    var scores = Object.create( null );
    // Helper variables for class, feature, it's value and weight.
    var c, f, v, w;
    // Previous class; finally contains the predicted class!
    var pc = '';
    // Previous value.
    var pv = -Infinity;

    // Compute scores for each class; will add bias later when we just loop for
    // classes while finding the maximum score.
    for ( f in features ) {
      w = specificWeights[ f ];
      v = features[ f ];
      // Check if weights for that featues exist and feature value is **non-zero**.
      if ( w && v ) {
        for ( c in specificWeights[ f ] ) {
          scores[ c ] = ( scores[ c ] || 0 ) + ( v * w[ c ] );
        }
      }
    }
    // Find the best class with the maximum score.
    for ( c in scores ) {
      // Add bias at this stage.
      v = scores[ c ] + specificBiases[ c ];
      if ( v > pv ) {
        pc = c;
        pv = v;
      } else if ( v === pv && c > pc ) {
        // Values being equal, fall back to alpha sort!
        pc = c;
      }
    }

    return ( pc || 'unknown' );
  }; // predictUsingSpecificWeights()

  var adjustWt = function ( f, v, c ) {
    var w = ( weights[ f ][ c ] || 0 );
    lastWtUpdatedAt[ f ][ c ] = lastWtUpdatedAt[ f ][ c ] || 0;
    // Update sum by adding the last weight times the difference between last
    // and current update counts.
    sumOfWts[ f ][ c ] = ( sumOfWts[ f ][ c ] || 0 ) + ( ( updates - lastWtUpdatedAt[ f ][ c ] ) * w );
    lastWtUpdatedAt[ f ][ c ] = updates;
    weights[ f ][ c ] = w + v;
  }; // adjustWt()

  var adjustBs = function ( v, c ) {
    var b = ( biases[ c ] || 0 );
    // Update sum by adding the last weight times the difference between last
    // and current update counts.
    sumOfBiases[ c ] = ( sumOfBiases[ c ] || 0 ) + ( ( updates - lastBsUpdatedAt[ c ] ) * b );
    lastBsUpdatedAt[ c ] = updates;
    biases[ c ] = b + v;
  }; // adjustBs()

  var averageBalance = function () {
    var c, f;
    var b, w;
    // Follows the logic similar to `adjustWt()` & `adjustBs()`.
    // Process weights.
    for ( f in weights ) {
      for ( c in weights[ f ] ) {
        w = weights[ f ][ c ];
        sumOfWts[ f ][ c ] = ( sumOfWts[ f ][ c ] || 0 ) + ( ( updates - lastWtUpdatedAt[ f ][ c ] ) * w );
        // Compute average of weights.
        avgWts[ f ][ c ] = +( sumOfWts[ f ][ c ] / updates ).toFixed( 3 );
      }
    }
    // Process biases.
    for ( c in biases ) {
      b = biases[ c ];
      sumOfBiases[ c ] = ( sumOfBiases[ c ] || 0 ) + ( ( updates - lastBsUpdatedAt[ c ] ) * b );
      // Compute average of biases.
      avgBiases[ c ] = +( sumOfBiases[ c ] / updates ).toFixed( 3 );
    }
  }; // averageBalance()

  var adjustWeights = function ( data, guess ) {
    var features = data[ 0 ];
    var truth = data[ 1 ].label;
    // Helper variable for feature.
    var f, v;
    // Next update means increment updates.
    updates += 1;
    for ( f in features ) {
      if ( !weights[ f ] ) weights[ f ] = Object.create( null );
      if ( !lastWtUpdatedAt[ f ] ) lastWtUpdatedAt[ f ] = Object.create( null );
      if ( !sumOfWts[ f ] ) sumOfWts[ f ] = Object.create( null );

      v = features[ f ];
      adjustWt( f, v, truth );
      if ( guess !== 'unknown' ) adjustWt( f, -v, guess );
    }
    adjustBs( +1, truth );
    if ( guess !== 'unknown' ) adjustBs( -1, guess );
  }; // adjustWeights()

  var shuffle = function ( arr ) {
    if ( shuffleData ) shuffleArray( arr );
  };

  var learnFromData = function ( data ) {
    // Prediction.
    var guess;
    // Helper variables for loops.
    var j, k;

    // Starting from **1** ensures that we iterate **maxIterations** times.
    for ( j = 0; j < maxIterations; j += 1 ) {
      for ( k = 0; k < data.length; k += 1 ) {
        guess = predictUsingSpecificWeights( data[ k ][ 0 ], weights, biases );
        if ( guess !== data[ k ][ 1 ].label ) adjustWeights( data[ k ], guess );
      } // for data.length
      // Random shuffle of the data — critical for perceptron learning.
      shuffle( data );
    } // for maxIterations

    averageBalance();
  }; // learnFromData()

  var learnFromExtractedFeatures = function ( data ) {
    var features;
    // Prediction.
    var guess;
    // Helper variables for loops.
    var j, k, l;

    // Starting from **1** ensures that we iterate **maxIterations** times.
    for ( j = 0; j < maxIterations; j += 1 ) {
      for ( k = 0; k < data.length; k += 1 ) {
        features = featureExtractor( data[ k ] );
        for ( l = 0; l < features.length; l += 1 ) {
          guess = predictUsingSpecificWeights( features[ l ][ 0 ], weights, biases );
          if ( guess !== features[ l ][ 1 ].label ) adjustWeights( features[ l ], guess );
        } // for features.length
      } // for data.length
      // Random shuffle of the data — critical for perceptron learning.
      shuffle( data );
    } // for maxIterations

    averageBalance();
  }; // learnFromExtractedFeatures()

  // ### learn
  /**
   *
   * Learns from the **examples**. The hyperparameters, defined via [`defineConfig`](#defineConfig),
   * control learning process.
   *
   * @method Perceptron#learn
   * @param {array[]} examples each example is a 2-element array. The
   * first element describes example's features and the second one defines
   * its class label. Both of these are expressed in form of an object. The
   * features object contains **name/numeric-value** pairs for every feature, whereas the
   * class label contains single name/string-value pair as `{ label: <class> }`.
   *
   * @return {number} Number of examples passed.
   * @example
   * myPerceptron.learn( examples );
   * @throws Error if all `examples` belong to only **one** class OR if attempted
   * after [`importJSON()`](#importJSON).
  */
  var learn = function ( examples ) {
    if ( imported ) {
      throw Error( 'wink-perceptron: learnings already imported.' );
    }

    if ( typeof featureExtractor === 'function' ) {
      learnFromExtractedFeatures( examples );
    } else {
      learnFromData( examples );
    }

    if ( ( Object.keys( sumOfBiases ) ).length < 2 ) {
      throw Error( 'wink-perceptron: there must be at least 2 classes in examples.' );
    }

    examplesSeen = examples.length;
    return ( examplesSeen );
  }; // learn()

  // ### predict
  /**
   *
   * Predicts the label for the input `features`. If it is unable to predict then
   * it returns a value **`unknown`**.
   *
   * @method Perceptron#predict
   * @param {object} features object that contains **name/value** pairs for every
   * feature.
   *
   * @return {string} Predicted class label for the input `features`.
   * @example
   * myPerceptron.predict( features );
   * @throws Error if prediction is attempted without [learning](#learn) or [import](#importJSON).
  */
  var predict = function ( features ) {
    if ( !imported && examplesSeen === 0 ) {
      throw Error( 'wink-perceptron: prediction is not possible without learning!' );
    }

    // Use averaged weights.
    return predictUsingSpecificWeights( features, avgWts, avgBiases );
  }; // predict()

  // ### defineConfig
  /**
   *
   * Defines the hyperparameters for perceptron.
   *
   * @method Perceptron#defineConfig
   * @param {object} config table below details the properties of `config` object.
   *
   * *An empty config object restores the default configuration.*
   *
   * @param {boolean} [config.shuffleData=false] determines whether or not the
   * training examples should be randomly shuffled after each iteration a.k.a epoch.
   * @param {number} [config.maxIterations=9] number of passes that must be made
   * over the examples in order to complete the learning.
   * @param {function} [config.featureExtractor=null] extracts feature(s) along with the corresponding class label(s)
   * from example prior to each iteration. This is useful when raw examples need to be
   * passed to [`learn()`](#learn) instead of features & labels. If it extracts >1 features
   * then each of the extracted feature/label pair is processed sequentially during learning.
   * Note `shuffleData` value will only control the shuffling of input examples and
   * not of the extracted features with this function.
   * @return {object} A copy of configuration defined.
   * @example
   * // Enable random shuffling of examples!
   * myPerceptron.defineConfig( { shuffleData: true } );
   * // -> { shuffleData: true, maxIterations: 9, featureExtractor: null }
  */
  var defineConfig = function ( config ) {
    if ( !helpers.object.isObject( config ) ) {
      throw Error( 'wink-perceptron: config must be an object, instead found: ' + ( typeof config ) );
    }
    // Convert 'truthy -> true' or `falsy -> false`. This also implies that
    // default is **`false`**.
    shuffleData = !!config.shuffleData;
    // Default # of maximum iteration is **6**.
    maxIterations = config.maxIterations || maxIterations;
    if ( maxIterations < 1 ) {
      throw Error( 'wink-perceptron: maxIterations should be >1' );
    }
    // Ordered Set Of Features Extractor function; default is none!
    featureExtractor = ( config.featureExtractor === undefined ) ? featureExtractor : config.featureExtractor;
    if ( ( featureExtractor !== null ) && ( typeof featureExtractor !== 'function' )  ) {
      throw Error( 'wink-perceptron: featureExtractor must be a function, instead found: ' + ( typeof featureExtractor ) );
    }

    return ( { shuffleData: shuffleData, maxIterations: maxIterations, featureExtractor: featureExtractor } );
  }; // defineConfig()

  // ### reset
  /**
   * It completely resets the perceptron by re-initializing all the learning
   * related variables but does not touch the existing configuration.
   *
   * *Since it does not reset the existing configuration, user must define it
   * again prior to learning if required.*
   *
   * @method Perceptron#reset
   * @return {boolean} Always true.
   * @example
   * myPerceptron.reset();
   * // -> true
  */
  var reset = function () {
    // Initialize learning variables.
    weights = Object.create( null );
    biases = Object.create( null );
    sumOfWts = Object.create( null );
    sumOfBiases = Object.create( null );
    lastWtUpdatedAt = Object.create( null );
    lastBsUpdatedAt = Object.create( null );
    updates = 0;
    examplesSeen = 0;
    imported = false;
    // Setup aliases.
    avgWts = sumOfWts;
    avgBiases = sumOfBiases;
    // Always true!
    return true;
  }; // reset()

  // ### exportJSON
  /**
   * Exports the learning as a JSON, which may be saved as a text file for
   * later use via [`importJSON()`](#importJSON).
   *
   * @method Perceptron#exportJSON
   * @return {string} Learning in JSON format.
   * @example
   * // Assuming that learn() method has been already succesful.
   * myPerceptron.exportJSON();
   * // -> JSON string.
   * @throws Error if export is attempted without [learning](#learn).
  */
  var exportJSON = function ( ) {
    if ( examplesSeen === 0 ) {
      throw Error( 'wink-perceptron: nothing to export, learning is a prerequisite!' );
    }

    return (
      JSON.stringify( [
        avgWts,
        avgBiases,
        // For future expansion...
        {},
        []
      ] )
    );
  }; // exportJSON()

  // ### importJSON
  /**
  * Imports an existing JSON learning for prediction purpose **only**; it cannot
  * be used for further [learning](#learn).
  *
  * @method Perceptron#importJSON
  * @param {JSON} json containing learnings in as exported by [`exportJSON`](#exportjson).
  * @return {boolean} Always true.
  * @example
  * // Assuming that `json` already has a valid JSON string.
  * myPerceptron.importJSON( json );
  * @throws Error if `json` is invalid.
  */
  var importJSON = function ( json ) {
   var parsedJSON;
   if ( !json ) {
     throw Error( 'wink-perceptron: undefined or null JSON encountered, import failed!' );
   }
   // Validate json format
   var isOK = [
     helpers.object.isObject,
     helpers.object.isObject,
     helpers.object.isObject,
     helpers.array.isArray
   ];

   try {
     parsedJSON = JSON.parse( json );
   } catch ( ex ) {
     throw Error( 'wink-perceptron: JSON parsing error during import:\n\t' + ex.message );
   }

   if ( !helpers.array.isArray( parsedJSON ) || parsedJSON.length !== isOK.length ) {
     throw Error( 'wink-perceptron: invalid JSON encountered, can not import.' );
   }
   for ( var i = 0; i < isOK.length; i += 1 ) {
     if ( !isOK[ i ]( parsedJSON[ i ] ) ) {
       throw Error( 'wink-perceptron: invalid JSON encountered, can not import.' );
     }
   }
   // All good, setup variable values.
   // First reset everything.
   reset();
   // Load variable values.
   avgWts = sumOfWts = parsedJSON[ 0 ];
   avgBiases = sumOfBiases = parsedJSON[ 1 ];
   // Return success.
   imported = true;
   return true;
  }; // importJSON()

  methods.defineConfig = defineConfig;
  methods.learn = learn;
  methods.predict = predict;
  methods.reset = reset;
  methods.exportJSON = exportJSON;
  methods.importJSON = importJSON;
  // methods.show = function () { console.log(sumOfWts); console.log(sumOfBiases); console.log( updates ); }; // eslint-disable-line
  return ( methods );
}; // perceptron()

module.exports = perceptron;
