import React from 'react';
import api from './lib/api';
import ndarray from 'ndarray';

export default class Map extends React.Component {

  componentWillMount(){
    this.fetch();
  }

  fetch( bbox ) {
    const z = 18;
    const x = 50355;
    const y = 106303;
    let url = 'https://a.tiles.mapbox.com/v4/mapbox.streets-satellite/';
    url += `${z}/${x}/${y}.png?access_token=pk.eyJ1IjoiY2hlbG0iLCJhIjoiY2lyNjk0dnJiMDAyNGk5bmZnMTk4dDNnaiJ9.BSE3U0yfeyD6jtSf4t8xzQ`;

    const img = new Image()
    img.crossOrigin = "Anonymous"
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = img.width
      canvas.height = img.height
      const context = canvas.getContext('2d')
      context.drawImage(img, 0, 0)
      const pixels = context.getImageData(0, 0, img.width, img.height);
      console.log(pixels);
      const arr = ndarray(new Uint8Array(pixels.data), [img.width, img.height, 4], [4, 4*img.width, 1], 0)
      console.log(arr);
    }
    img.onerror = function(err) {
      console.log('err', err)
    }
    img.src = url
  }

  render() {
    return <div>FOO</div>;
  }
}
