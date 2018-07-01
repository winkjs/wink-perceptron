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

// Extracts features from a single row of data.
var extractFeatures = function ( e ) {
 return ( [ [ { sl: +e[ 0 ], sw: +e[ 1 ], pl: +e[ 2 ], pw: +e[ 3 ] }, { label: e[ 4 ] } ] ] );
}; // extractFeatures()

// IRIS Data is sourced from UC Irvine Machine Learning Repository;
// source: https://archive.ics.uci.edu/ml/datasets/iris
// Prepare data:
lines = fs.readFileSync( './test/data/iris-data-train.csv', 'utf8' ).split( '\n' );
lines.pop();
td = [];
lines.forEach( function ( e ) {
  td.push( e.split( ',' ) );
} );

rawTrainingData = td.slice( 0 );
td.forEach( function ( e ) {
  trainingData.push( extractFeatures( e )[ 0 ] );
} );

lines = fs.readFileSync( './test/data/iris-data-test.csv', 'utf8' ).split( '\n' );
lines.pop();
td = [];
lines.forEach( function ( e ) {
  td.push( e.split( ',' ) );
} );
td.forEach( function ( e ) {
  testData.push( extractFeatures( e )[ 0 ] );
} );

// Tests
describe( 'instantiate perceptron ', function () {
  it( 'must return 3 methods', function () {
    expect( Object.keys( perceptron() ).length ).to.equal( 3 );
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

  it( 'must predict with >90% accuracy test data', function () {
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
    expect( p.defineConfig( { maxIterations: 210, featureExtractor: extractFeatures } ) )
      .to.deep.equal( { shuffleData: false, maxIterations: 210, featureExtractor: extractFeatures } );
  } );

  it( 'learn must return 120', function () {
    expect( p.learn( rawTrainingData ) ).to.equal( 120 );
  } );

  it( 'must predict with >90% accuracy test data', function () {
    var pass = 0;
    testData.forEach( function ( e ) {
      if ( p.predict( e[ 0 ] ) === e[ 1 ].label ) pass += 1;
    } );
    expect( pass / testData.length > 0.9 ).to.equal( true );
  } );
} );
