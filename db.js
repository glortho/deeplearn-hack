import localforage from 'localforage';

localforage.config({
  name:      'mlmap',
  version:   1.0,
  storeName: 'training_data' // Should be alphanumeric, with underscores.
});

export default localforage;
