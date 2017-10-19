import localforage from 'localforage';

localforage.config({
  name:      'mlmap',
  version:   1.0,
  storeName: 'training_data' // Should be alphanumeric, with underscores.
});

export const addTraining = ( bbox, label ) =>
  localforage.setItem( JSON.stringify( bbox ), label );

export const getTrainingData = () => new Promise( resolve => {
  const map = new Map();
  localforage.iterate(( label, bbox ) => {
    map.set( JSON.parse( bbox ), label );
  } , () => resolve( map ));
});

export const removeTraining = ( bbox ) =>
  localforage.removeItem( JSON.stringify( bbox ) );

export const clear = localforage.clear;

window.db = {
  addTraining,
  clear,
  getTrainingData,
  removeTraining,
  _db: localforage
};
