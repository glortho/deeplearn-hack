import { FeatureGroup, Map as LeafletMap, TileLayer } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import React from 'react';
import api from './lib/api';
import ndarray from 'ndarray';
import SphericalMercator from 'sphericalmercator';

const merc = new SphericalMercator({
  size: 256
});

import model, { train } from './model';
import db from './db';

export default class Map extends React.Component {

  componentWillMount() {
    let link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet/1.2.0/leaflet.css";
    document.getElementsByTagName("head")[0].appendChild(link);

    link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet.draw/0.4.12/leaflet.draw.css";
    document.getElementsByTagName("head")[0].appendChild(link);

  }

  fetch = ( bbox ) => {
    const {minX, minY, maxX, maxY } = merc.xyz(bbox, 18);
    console.log( minX, minY, maxX, maxY)
    for ( let x=minX; x < maxX + 1; x++ ) {
      for ( let y=minY; y < maxY + 1; y++ ) {
        console.log(x,y)

        // convert to zxy
        const z = 18;
        //const x = 50355;
        //const y = 106303;
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
          console.log(pixels.data[0]);
          //const arr = ndarray(new Uint8Array(pixels.data), [img.width, img.height, 4], [4, 4*img.width, 1], 0)
          model.addTraining( Array.from(pixels.data), 1 );
          train();
        }
        img.onerror = function(err) {
          console.log('err', err)
        }
        img.src = url
      }
    }
  }

  mapOptions = {
    style: {
      height: "auto",
      position: "absolute",
      top: 0,
      left: 0,
      right: 0,
      bottom: 0
    }
  }

  shapeOptions = {
    color: "#01bf0a",
    weight: 3,
    fill: false

  }

  train = event => {
    this.fetch( event.layer._bounds.toBBoxString().split(',').map( c => parseFloat(c) ) );
  }

  render() {
    return (
      <div>
        <LeafletMap ref="map" center={ [32.175068, -110.851364 ] } zoom={18} { ...this.mapOptions }>
          <TileLayer
            url="https://{s}.tiles.mapbox.com/v4/mapbox.streets-satellite/{z}/{x}/{y}@2x.png?access_token=pk.eyJ1IjoiY2hlbG0iLCJhIjoiY2lyNjk0dnJiMDAyNGk5bmZnMTk4dDNnaiJ9.BSE3U0yfeyD6jtSf4t8xzQ"
            attribution = "&copy; Mapbox | &copy; DigitalGlobe"
          />
          <FeatureGroup>
            <EditControl
              position='topleft'
              onCreated={ this.train }
              draw={{
                polyline: false,
                polygon: this.shapeOptions ,
                circle: false,
                point: false,
                marker: false,
                circlemarker: false,
                rectangle: this.shapeOptions ,
                edit: true
              }}
            />
          </FeatureGroup>

        </LeafletMap>
      </div>
    );
  }
}
