import { FeatureGroup, Map as LeafletMap, TileLayer } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import React from 'react';
import api from './lib/api';
import ndarray from 'ndarray';
import imshow from 'ndarray-imshow';
import SphericalMercator from 'sphericalmercator';
//import { Array3D } from 'deeplearn';
import unpack from 'ndarray-unpack';

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

  unpack_flat = (view) => unpack(view)
    .reduce((master, row) => master.concat( [ ...row ] ), [] );

  tilePx = (lat, lon, tile_bbox) => {
    const w = 256;
    const h = 256;
    const lat0 = tile_bbox[3];
    const lon0 = tile_bbox[0];
    const latD = tile_bbox[3] - tile_bbox[1];
    const lonD = tile_bbox[2] - tile_bbox[0];
    let longitude = lon;
    longitude -= lon0;
    let latitude = lat0 - lat;
    const x = (w*(longitude/lonD));
    const y = Math.min(256.0, (h*(latitude/latD)));
    return [parseInt(Math.max(0,x)), parseInt(Math.max(0,y))];
  } 

  fetch = ( bbox ) => {
    const z = 18;
    const {minX, minY, maxX, maxY } = merc.xyz(bbox, z);
    for ( let x=minX; x < maxX + 1; x++ ) {
      for ( let y=minY; y < maxY + 1; y++ ) {
        let url = 'https://a.tiles.mapbox.com/v4/mapbox.streets-satellite/';
        url += `${z}/${x}/${y}.png?access_token=pk.eyJ1IjoiY2hlbG0iLCJhIjoiY2lyNjk0dnJiMDAyNGk5bmZnMTk4dDNnaiJ9.BSE3U0yfeyD6jtSf4t8xzQ`;
        console.log(url)

        const tile_bbox = merc.bbox(x, y, z);
        const minXY = this.tilePx(bbox[3], bbox[0], tile_bbox);
        const maxXY = this.tilePx(bbox[1], bbox[2], tile_bbox);

        const img = new Image()
        img.crossOrigin = "Anonymous"
        img.onload = () => {
          const canvas = document.createElement('canvas')
          canvas.width = img.width
          canvas.height = img.height
          const context = canvas.getContext('2d')
          context.drawImage(img, 0, 0)
          const pixels = context.getImageData(0, 0, img.width, img.height);

          const arr = ndarray(new Uint8Array(pixels.data), [img.width, img.height, 4], [4*img.width, 4, 1], 0);
          if ( (maxXY[1] - minXY[1]) > 0 && ( maxXY[0] - minXY[0] ) > 0 ) {
            const miny = minXY[1];
            const minx = minXY[0];
            const maxy = maxXY[1];
            const maxx = maxXY[0];
            const clip = arr
              .hi(maxy, maxx)
              .lo(miny, minx)

            const red = this.unpack_flat(clip.pick(null, null, 0));
            const green = this.unpack_flat(clip.pick(null, null, 1));
            const blue = this.unpack_flat(clip.pick(null, null, 2));
            console.log(red)

            //const sliced = Array.from(pixels.data.slice(0, model.inputSize));
            //console.log(Array.from(sliced));

            //model.addTraining( sliced, 1.0 );
            //model.step = 0;
            //train(); 
          }
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
