import localforage from 'localforage';

localforage.config({
  name:      'mlmap',
  version:   1.0,
  storeName: 'training_data' // Should be alphanumeric, with underscores.
});

export const addTraining = ({ bbox, x, y, label, inference }) =>
  localforage.setItem( JSON.stringify({ bbox, x, y }), { inference, label });

export const getTrainingData = () => new Promise( resolve => {
  const map = new Map();
  localforage.iterate(( val, key ) => {
    map.set( JSON.parse( key ), val );
  } , () => resolve( map ));
});

export const removeTraining = key =>
  localforage.removeItem( JSON.stringify( key ) );

export const clear = localforage.clear;

window.db = {
  addTraining,
  clear,
  getTrainingData,
  removeTraining,
  _db: localforage
};
