- Build docker image first
- Run docker image:
  ~~~~
  docker run --rm -v .:/app -i -t minsdr /bin/bash
  ~~~~
- Create dataset: 
  ~~~~
  python3 generate_RML2016.10a.py
  ~~~~
- Examples for reading dataset:
  - using h5py:
  ~~~~
  import numpy as np
  import h5py

  dataset = h5py.File('RML2016.10a_dict.h5', 'r')
  print(list(dataset.keys()))

  ra = dataset['QAM64_18'][0]
  ~~~~
  - using cPickle:
  ~~~~
  import numpy as np
  import _pickle as cPickle

  file = open("RML2016.10a_dict.dat", "rb");
  dataset = cPickle.load( file )

  ra = dataset['QAM64', 18][0]
  ~~~~