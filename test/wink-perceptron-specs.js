//     wink-perceptron
//     Multi-class averaged perceptron
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
/* eslint-disable no-console */
/* eslint-disable no-sync */
var fs = require( 'fs' );
var lines;
var trainingData = [];
var testData = [];
var rawTrainingData = [];
var td;


var chai = require( 'chai' );
var mocha = require( 'mocha' );
var perceptron = require( '../src/wink-perceptron.js' );

var expect = chai.expect;
var describe = mocha.describe;
var it = mocha.it;

const METHODS = 6;

// Extracts features from a single row of data.
var extractFeatures = function ( e ) {
 return ( [ [ { sl: +e[ 0 ], sw: +e[ 1 ], pl: +e[ 2 ], pw: +e[ 3 ] }, { label: e[ 4 ] } ] ] );
}; // extractFeatures()

// Text Data.
var intents = [
  [ { i: 1, need: 1, loan: 1, for: 1, a: 1, new: 1, car: 1 }, { label: 'autoloan' } ],
  [ { i: 1, need: 1, to: 1, borrow: 1, for: 1, a: 1, new: 1, car: 1 }, { label: 'autoloan' } ],
  [ { i: 1, would: 1, like: 1, to: 1, foreclose: 1, my: 1, loan: 1 }, { label: 'prepay' } ],
  [ { i: 1, want: 1, to: 1, prepay: 1, my: 1, loan: 1 }, { label: 'prepay' } ]
];

// IRIS Data is sourced from UC Irvine Machine Learning Repository;
// source: https://archive.ics.uci.edu/ml/datasets/iris
// Prepare data:
// Training data preparation.
lines = fs.readFileSync( './test/data/iris-data-train.csv', 'utf8' ).split( '\n' );
lines.pop();
td = [];
lines.forEach( function ( e ) {
  td.push( e.split( ',' ) );
} );
// The `rawTrainingData` will now be a copy of `td` array.
rawTrainingData = td.slice( 0 );
td.forEach( function ( e ) {
  trainingData.push( extractFeatures( e )[ 0 ] );
} );

// Test data preparation.
lines = fs.readFileSync( './test/data/iris-data-test.csv', 'utf8' ).split( '\n' );
lines.pop();
td = [];
lines.forEach( function ( e ) {
  td.push( e.split( ',' ) );
} );
td.forEach( function ( e ) {
  testData.push( extractFeatures( e )[ 0 ] );
} );

// Sample test data.
const sampleTestData = {
  setosa: { sepalLength: 4.9, sepalWidth: 3, petalLength: 1.4, petalWidth: 0.2 },
  versicolor: { sepalLength: 6.4, sepalWidth: 3.2, petalLength: 4.5, petalWidth: 1.5 },
  virginica: { sepalLength: 7.2, sepalWidth: 3.6, petalLength: 6.1, petalWidth: 2.5 }
};

var learnings = require( './data/learnings.json' );
learnings = JSON.stringify( learnings );

// Tests
describe( 'instantiate perceptron', function () {
  it( 'must return ' + METHODS + ' methods', function () {
    expect( Object.keys( perceptron() ).length ).to.equal( METHODS );
  } );
} );

describe( 'defineConfig', function () {
  it( 'should throw error if config is not passed', function () {
      expect( perceptron().defineConfig.bind( undefined, undefined ) ).to.throw( 'wink-perceptron: config must be an object, instead found:' );
  } );

  it( 'should throw error maxIterations <1', function () {
      expect( perceptron().defineConfig.bind( undefined, { maxIterations: -1 } ) ).to.throw( 'wink-perceptron: maxIterations should be >1' );
  } );

  it( 'should throw error featureExtractor must be a function', function () {
      expect( perceptron().defineConfig.bind( undefined, { featureExtractor: 1 } ) ).to.throw( 'wink-perceptron: featureExtractor must be a function, instead found:' );
  } );

  it( 'should default configuration with empty config input', function () {
      expect( perceptron().defineConfig( {} ) )
      .to.deep.equal( { shuffleData: false, maxIterations: 9, featureExtractor: null } );
  } );
} );

describe( 'train & predict using extracted features from iris data', function () {
  var p = perceptron();

  it( 'defineConfig must return the config in force', function () {
    expect( p.defineConfig( { shuffleData: false, maxIterations: 210 } ) )
      .to.deep.equal( { shuffleData: false, maxIterations: 210, featureExtractor: null } );
  } );

  it( 'learn must return 120', function () {
    expect( p.learn( trainingData ) ).to.equal( 120 );
  } );

  it( 'must predict with >90% accuracy in test data', function () {
    var pass = 0;
    testData.forEach( function ( e ) {
      if ( p.predict( e[ 0 ] ) === e[ 1 ].label ) pass += 1;
    } );
    expect( pass / testData.length > 0.9 ).to.equal( true );
  } );
} );

describe( 'train & predict using from raw iris data', function () {
  var p = perceptron();

  it( 'defineConfig must return the config in force', function () {
    expect( p.defineConfig( { maxIterations: 21, featureExtractor: extractFeatures, shuffleData: true } ) )
      .to.deep.equal( { shuffleData: true, maxIterations: 21, featureExtractor: extractFeatures } );
  } );

  it( 'learn must return 120', function () {
    expect( p.learn( rawTrainingData ) ).to.equal( 120 );
  } );

  it( 'must predict with >90% accuracy in test data', function () {
    var pass = 0;
    testData.forEach( function ( e ) {
      if ( p.predict( e[ 0 ] ) === e[ 1 ].label ) pass += 1;
    } );
    expect( pass / testData.length > 0.9 ).to.equal( true );
  } );

  it( 'exported learnings must be an array of length 4', function () {
    expect( ( JSON.parse( p.exportJSON() ).length ) ).to.equal( 4 );
  } );
} );

describe( 'train & predict intent using from text data', function () {
  var p = perceptron();

  it( 'defineConfig must return the config in force', function () {
    expect( p.defineConfig( { shuffleData: true } ) )
      .to.deep.equal( { shuffleData: true, maxIterations: 9, featureExtractor: null } );
  } );

  it( 'learn must return 4', function () {
    expect( p.learn( intents ) ).to.equal( 4 );
  } );

  it( 'must predict autoloan', function () {
    expect( p.predict( { need: 1, to: 1, borrow: 1, money: 1, for: 1, a: 1, new: 1, vehicle: 1 } ) ).to.equal( 'autoloan' );
  } );
} );

describe( 'train & predict for tie in class scores', function () {
  var p = perceptron();
  var data = [
    [ { bad: 1 }, { label: 'L0' } ],
    [ { good: 1 }, { label: 'L1' } ],
    [ { bad: 1, good: 1 }, { label: 'L1' } ]
  ];

  it( 'defineConfig must return the config in force', function () {
    expect( p.defineConfig( { shuffleData: false, maxIterations: 1 } ) )
      .to.deep.equal( { shuffleData: false, maxIterations: 1, featureExtractor: null } );
  } );

  it( 'learn must return 3', function () {
    expect( p.learn( data ) ).to.equal( 3 );
  } );

  it( 'must predict label L0', function () {
    expect( p.predict( { bad: 1, good: 1 } ) ).to.equal( 'L0' );
  } );
} );

describe( 'reset must unlearn every thing', function () {
  var p = perceptron();

  it( 'defineConfig must return the config in force', function () {
    expect( p.defineConfig( { maxIterations: 21, featureExtractor: extractFeatures, shuffleData: true } ) )
      .to.deep.equal( { shuffleData: true, maxIterations: 21, featureExtractor: extractFeatures } );
  } );

  it( 'learn must return 120', function () {
    expect( p.learn( rawTrainingData ) ).to.equal( 120 );
  } );

  it( 'must predict with >90% accuracy in test data', function () {
    var pass = 0;
    testData.forEach( function ( e ) {
      if ( p.predict( e[ 0 ] ) === e[ 1 ].label ) pass += 1;
    } );
    expect( pass / testData.length > 0.9 ).to.equal( true );
  } );

  it( 'reset must return true', function () {
    expect( p.reset( ) ).to.equal( true );
  } );

  it( 'must throw error on prediction without learning after reset', function () {
    expect( perceptron().predict.bind( undefined, testData[ 0 ][ 0 ] ) )
      .to.throw( 'wink-perceptron: prediction is not possible without learning!' );
  } );

  it( 'defineConfig must return the newly set config', function () {
    expect( p.defineConfig( { shuffleData: true, maxIterations: 9, featureExtractor: null } ) )
      .to.deep.equal( { shuffleData: true, maxIterations: 9, featureExtractor: null } );
  } );

  it( 'learn must return 4', function () {
    expect( p.learn( intents ) ).to.equal( 4 );
  } );

  it( 'must predict autoloan', function () {
    expect( p.predict( { need: 1, to: 1, borrow: 1, money: 1, for: 1, a: 1, new: 1, vehicle: 1 } ) ).to.equal( 'autoloan' );
  } );

  it( 'must throw error on learning with examples containing single class after reset', function () {
    expect( p.reset( ) ).to.equal( true );
    expect( perceptron().learn.bind( undefined, [ [ { i: 1, need: 1, loan: 1, for: 1, a: 1, new: 1, car: 1 }, { label: 'autoloan' } ] ] ) )
      .to.throw( 'wink-perceptron: there must be at least 2 classes in examples' );
  } );
} );

describe( 'predict using imported learnings of iris data', function () {
  var p = perceptron();

  it( 'must return true on successful import', function () {
    expect( p.importJSON( learnings ) ).to.equal( true );
  } );

  it( 'must predict with sample test data correctly', function () {
    expect( p.predict( sampleTestData.setosa ) ).to.equal( 'Iris-setosa' );
    expect( p.predict( sampleTestData.versicolor ) ).to.equal( 'Iris-versicolor' );
    expect( p.predict( sampleTestData.virginica ) ).to.equal( 'Iris-virginica' );
  } );

  it( 'must throw error on learning after import', function () {
    expect( p.learn.bind( undefined, [ [ { i: 1, need: 1, loan: 1, for: 1, a: 1, new: 1, car: 1 }, { label: 'autoloan' } ] ] ) )
      .to.throw( 'wink-perceptron: learnings already imported.' );
  } );

  it( 'must throw error on importing without json', function () {
    expect( p.importJSON.bind( undefined, undefined ) )
      .to.throw( 'wink-perceptron: undefined or null JSON encountered, import failed!' );
  } );

  it( 'must throw error on importing non-string json', function () {
    expect( p.importJSON.bind( undefined, { x: 3 } ) )
      .to.throw( 'wink-perceptron: JSON parsing error during import:' );
  } );

  it( 'must throw error on importing invalid json', function () {
    expect( p.importJSON.bind( undefined, '3' ) )
      .to.throw( 'wink-perceptron: invalid JSON encountered, can not import.' );

    expect( p.importJSON.bind( undefined, '[ [], {}, 2, 3 ]' ) )
      .to.throw( 'wink-perceptron: invalid JSON encountered, can not import.' );
  } );
} );
