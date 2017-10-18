import { FeatureGroup, Map as LeafletMap } from 'react-leaflet';
import { EditControl } from 'react-leaflet-draw';
import React from 'react';

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
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet/1.2.0/leaflet.css";
    document.getElementsByTagName("head")[0].appendChild(link);

    link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "//cdnjs.cloudflare.com/ajax/libs/leaflet.draw/0.4.12/leaflet.draw.css";
    document.getElementsByTagName("head")[0].appendChild(link);
  }
  render() {
    return (
      <div>
        <LeafletMap ref="map" { ...this.mapOptions }>
          <FeatureGroup>
            <EditControl
              position='topleft'
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
