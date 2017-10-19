import localforage from 'localforage';

localforage.config({
  name:      'mlmap',
  version:   1.0,
  storeName: 'training_data' // Should be alphanumeric, with underscores.
});

export default localforage;

window.db = localforage;

export const addTraining = ( bbox, label ) =>
  localforage.setItem( JSON.stringify( bbox ), label );
