const path = require( 'path' );
const webpack = require( 'webpack' );

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
  plugins: [
    new webpack.NamedModulesPlugin(),
    new webpack.HotModuleReplacementPlugin()
  ],
  entry: { bundle: path.join( __dirname, 'index.js' ) },
  output: {
    path: path.join( __dirname, 'public' )
  },
  devServer: {
    contentBase: path.join( __dirname, 'public' ),
    hot: true
  }
}
