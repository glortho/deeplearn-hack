import { Popup, Rectangle, FeatureGroup, Map as LeafletMap, TileLayer } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import React from 'react';
import ndarray from 'ndarray';
import imshow from 'ndarray-imshow';
import SphericalMercator from 'sphericalmercator';
import unpack from 'ndarray-unpack';
import { getTrainingData, removeTraining } from './db';

const merc = new SphericalMercator({
  size: 256
});

import model from './model';
import db from './db';

export default class MapComponent extends React.Component {

  constructor( props ) {
    super( props );
    this.state = { label: 1, rectangles: new Map() };
  }

  componentWillMount() {
    let link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet/1.2.0/leaflet.css";
    document.getElementsByTagName("head")[0].appendChild(link);

    link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet.draw/0.4.12/leaflet.draw.css";
    document.getElementsByTagName("head")[0].appendChild(link);

    getTrainingData().then( data =>
      data.forEach((label, key) => {
        this.setState( state => ({
          rectangles: state.rectangles.set( key.bbox, label )
        }));
        this.fetchImg( key, label, { addToDb: false } )
      })
    );
  }

  clearAll = () => {
    model.clear();
    this.setState( state => ({ rectangles: new Map() }));
  }

  fetchImg = ({ bbox, x, y }, label, options) => {
    const z = 17;
    let url = 'https://a.tiles.mapbox.com/v4/mapbox.streets-satellite/';
    url += `${z}/${x}/${y}.png?access_token=pk.eyJ1IjoiY2hlbG0iLCJhIjoiY2lyNjk0dnJiMDAyNGk5bmZnMTk4dDNnaiJ9.BSE3U0yfeyD6jtSf4t8xzQ`;
    const img = new Image()
    img.crossOrigin = "Anonymous"
    img.onload = () => model.addTraining({ bbox, x, y, img, label, options });
    img.onerror = function(err) {
      console.log('err', err)
    }
    img.src = url
  }

  fetch = ( bbox, label, options ) => {
    const {minX, minY, maxX, maxY } = merc.xyz(bbox, 17);
    for ( let x=minX; x < maxX + 1; x++ ) {
      for ( let y=minY; y < maxY + 1; y++ ) {
        this.fetchImg({ bbox, x, y }, label, options);
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
    fill: true
  }

  getBbox = bbox => bbox.getBounds().toBBoxString().split(',').map( c => parseFloat(c) )

  train = event => {
    this.fetch( this.getBbox( event.layer ), this.state.label );
  }

  removeTrainingData = event => {
    event.layers.getLayers().forEach( layer => {
      const bbox = this.getBbox( layer );
      this.removeRect( bbox );
      const {minX, minY, maxX, maxY } = merc.xyz(bbox, 17);
      for ( let x=minX; x < maxX + 1; x++ ) {
        for ( let y=minY; y < maxY + 1; y++ ) {
          removeTraining({ bbox, x, y });
        }
      }
    });
  }

  removeRect = bbox => {
    if ( this.state.rectangles.get( bbox ) ) {
      this.setState( state => {
        state.rectangles.delete( bbox );
        return { rectangles: state.rectangles };
      });
    }
  }

  removeBbox = bbox => () => {
    this.removeRect( bbox );
    const {minX, minY, maxX, maxY } = merc.xyz(bbox, 17);
    for ( let x=minX; x < maxX + 1; x++ ) {
      for ( let y=minY; y < maxY + 1; y++ ) {
        removeTraining({ bbox, x, y });
      }
    }
  }

  setLabel = label => () => this.setState({ label });

  render() {
    return (
      <div>
        <div style={{ zIndex: 10000, position: 'absolute', top: '14px', right: '10px'}}>
          <button onClick={ this.clearAll }>Clear All</button>&nbsp;
          <button onClick={() => model.train()}>Train</button>
        </div>
        <div style={{ zIndex: 10000, position: 'absolute', left: '60px', top: '14px', background: 'rgba(255,255,255,0.3)', padding: '6px', borderRadius: '2px', color: '#111' }}>
          <label>
            <input name="label" type="radio" value="1" onChange={ this.setLabel( 1 ) } checked={ this.state.label === 1 }/>Airplane&nbsp;
          </label>
          <label>
            <input name="label" type="radio" value="0" onChange={ this.setLabel( 0 ) } checked={ this.state.label === 0 }/>Not Airplane
          </label>
        </div>
        <LeafletMap ref="map" center={ [32.175068, -110.851364 ] } zoom={18} { ...this.mapOptions }>
          <TileLayer
            url="https://{s}.tiles.mapbox.com/v4/mapbox.streets-satellite/{z}/{x}/{y}@2x.png?access_token=pk.eyJ1IjoiY2hlbG0iLCJhIjoiY2lyNjk0dnJiMDAyNGk5bmZnMTk4dDNnaiJ9.BSE3U0yfeyD6jtSf4t8xzQ"
            attribution = "&copy; Mapbox | &copy; DigitalGlobe"
          />
          <FeatureGroup key="1">
            <EditControl
              key="1"
              position='topleft'
              onCreated={ this.train }
              onDeleted={ this.removeTrainingData }
              draw={{
                polyline: false,
                polygon: false,
                circle: false,
                point: false,
                marker: false,
                circlemarker: false,
                rectangle: {
                  shapeOptions: { ...this.shapeOptions, color: this.state.label ? this.shapeOptions.color : 'red' }
                },
                edit: true
              }}
            />
          </FeatureGroup>
          {[ ...this.state.rectangles ].map(([ bbox, label], idx) => {
            return (
              <Rectangle key={ idx } bounds={[[bbox[1], bbox[0]], [bbox[3], bbox[2]]]} color={ label ? this.shapeOptions.color : 'red' }>
                <Popup>
                  <div>
                    <button onClick={ this.removeBbox( bbox ) }>Delete</button>
                  </div>
                </Popup>
              </Rectangle>
            );
          })}
        </LeafletMap>
      </div>
    );
  }
}
