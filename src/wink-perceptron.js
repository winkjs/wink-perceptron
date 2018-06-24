//     wink-perceptron
//     Language agnostic named entity recognizer
//
//     Copyright (C) 2017-18  GRAYPE Systems Private Limited
//
//     This file is part of “wink-perceptron”.
//
//     “wink-perceptron” is free software: you can redistribute it
//     and/or modify it under the terms of the GNU Affero
//     General Public License as published by the Free
//     Software Foundation, version 3 of the License.
//
//     “wink-perceptron” is distributed in the hope that it will
//     be useful, but WITHOUT ANY WARRANTY; without even
//     the implied warranty of MERCHANTABILITY or FITNESS
//     FOR A PARTICULAR PURPOSE.  See the GNU Affero General
//     Public License for more details.
//
//     You should have received a copy of the GNU Affero
//     General Public License along with “wink-perceptron”.
//     If not, see <http://www.gnu.org/licenses/>.

//
var helpers = require( 'wink-helpers' );
var shuffle = helpers.array.shuffle;
/* eslint-disable guard-for-in */
// wink multi-class averaged perceptron
var perceptron = function () {
  // The weights matrix with **features** x **classes** dimensions.
  var weights = Object.create( null );
  // Bias for every **class**.
  var biases = Object.create( null );
  // Following set of five variables are used to compute averages. The average
  // is computed by dividing the accumulated sums by updates.
  // Sum of weights used for comuting average weight.
  var sumOfWts = Object.create( null );
  // Sum of biases used for computing average bias.
  var sumOfBiases = Object.create( null );
  // Captures the last moment/epoch when an update of weight for a class/feature
  // combo has occurred. The moment/epoch here is nothing but the update number.
  var lastWtUpdatedAt = Object.create( null );
  // Captures the last iteration when an update of bias for a class occurred.
  var lastBsUpdatedAt = Object.create( null );
  // The number of updates.
  var updates = 0;

  // Configuration variables and their default values.
  // Maximum number of learning iterations.
  var maxIterations = 9;
  // True means data will be shuffled after every iteration.
  var shuffleData = false;
  // Ordered Set Of Features Extractor function.
  var osofExtractor = null;

  // Returns!
  var methods = Object.create( null );


  var predict = function ( features ) {
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
      w = weights[ f ];
      v = features[ f ];
      // Check if weights for that featues exist and feature value is **non-zero**.
      if ( w && v ) {
        for ( c in weights[ f ] ) {
          scores[ c ] = ( scores[ c ] || 0 ) + ( v * w[ c ] );
        }
      }
    }
    // Find the best class with the maximum score.
    for ( c in scores ) {
      // Add bias at this stage.
      v = scores[ c ] + biases[ c ];
      if ( v > pv ) {
        pc = c;
        pv = v;
      } else if ( v === pv ) {
        // Everything being equal, fall back to alpha sort!
        if ( c > pc ) pc = c;
      }
    }

    return ( pc );
  }; // predict()

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
        sumOfWts[ f ][ c ] = +( sumOfWts[ f ][ c ] / updates ).toFixed( 3 );
      }
    }
    // Process biases.
    for ( c in biases ) {
      b = biases[ c ];
      sumOfBiases[ c ] = ( sumOfBiases[ c ] || 0 ) + ( ( updates - lastBsUpdatedAt[ c ] ) * b );
      sumOfBiases[ c ] = +( sumOfBiases[ c ] / updates ).toFixed( 3 );
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
      if ( guess ) adjustWt( f, -v, guess );
    }
    adjustBs( +1, truth );
    if ( guess ) adjustBs( -1, guess );
  }; // adjustWeights()

  var learnFromData = function ( data ) {
    // Prediction.
    var guess;
    // Helper variables for loops.
    var j, k;

    // Starting from **1** ensures that we iterate **maxIterations** times.
    for ( j = 1; j < maxIterations; j += 1 ) {
      // Random shuffle of the data — critical for perceptron learning.
      if ( shuffleData ) shuffle( data );
      for ( k = 0; k < data.length; k += 1 ) {
        guess = predict( data[ k ][ 0 ] );
        if ( guess !== data[ k ][ 1 ].label ) adjustWeights( data[ k ], guess );
      }
    }

    averageBalance();
  }; // learnFromData()

  var learnFromOSOData = function ( data ) {
    var sof;
    // Prediction.
    var guess;
    // Helper variables for loops.
    var j, k, l;

    // Starting from **1** ensures that we iterate **maxIterations** times.
    for ( j = 1; j < maxIterations; j += 1 ) {
      // Random shuffle of the data — critical for perceptron learning.
      if ( shuffleData ) shuffle( data );
      for ( k = 0; k < data.length; k += 1 ) {
        sof = osofExtractor( data[ k ] );
        for ( l = 0; l < sof.length; l += 1 ) {
          guess = predict( sof[ l ][ 0 ] );
          if ( guess !== sof[ l ][ 1 ].label ) adjustWeights( sof[ l ], guess );
        }
      }
    }

    averageBalance();
  }; // learnFromOSOData()

  var learn = function ( data ) {
    if ( typeof osofExtractor === 'function' ) {
      learnFromOSOData( data );
    } else {
      learnFromData( data );
    }
  }; // learn()

  var defineConfig = function ( configuration ) {
    // Convert 'truthy -> true' or `falsy -> false`. This also implies that
    // default is **`false`**.
    shuffleData = !!configuration.shuffleData;
    // Default # of maximum iteration is **6**.
    maxIterations = configuration.maxIterations || maxIterations;
    // Ordered Set Of Features Extractor function; default is none!
    osofExtractor = configuration.osofExtractor || osofExtractor;
  }; // defineConfig()

  methods.defineConfig = defineConfig;
  methods.learn = learn;
  methods.predict = predict;
  methods.show = function () { console.log(sumOfWts); console.log(sumOfBiases); console.log( updates ); }; // eslint-disable-line
  return ( methods );
}; // perceptron()

module.exports = perceptron;
