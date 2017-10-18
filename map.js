import { FeatureGroup, Map as LeafletMap, TileLayer } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import React from 'react';

import db from './db';
import model from './model';

export default class Map extends React.Component {

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
  componentWillMount() {
    let link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//unpkg.com/leaflet@1.2.0/dist/leaflet.css";
    link.integrity = 'sha512-M2wvCLH6DSRazYeZRIm1JnYyh22purTM+FDB5CsyxtQJYeKq83arPe5wgbNmcFXGqiSH2XR8dT/fJISVA1r/zQ==';
    link.crossOrigin = '';
    document.getElementsByTagName("head")[0].appendChild(link);

    link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet.draw/0.4.12/leaflet.draw.css";
    document.getElementsByTagName("head")[0].appendChild(link);
  }

  train = event => {
    console.log( event );
    // db.setItem()
    // model.train()
  }

  render() {
    return (
      <div>
        <LeafletMap ref="map" { ...this.mapOptions }>
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
