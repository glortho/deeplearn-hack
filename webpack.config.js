const path = require( 'path' );

module.exports = {
  module: {
    rules: [{
      test: /\.js$/,
      exclude: /node_modules(?!webpack\-dev\-server)/,
      use: [{
        loader: 'babel-loader'
      }]
    }]
  },
  entry: { bundle: path.join( __dirname, 'index.js' ) },
  output: {
    path: path.join( __dirname, 'public' )
  },
  devServer: {
    contentBase: path.join( __dirname, 'public' )
  }
}
