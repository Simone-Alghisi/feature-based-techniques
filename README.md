# Feature-based-techniques
Repository about feature-based techniques required in the second Computer Vision assignment.

## Running the algorithm
The file can be executed as it follows:

```console
$ python alghisi_simone_229355.py ALGORITHM /path/to/video
```

where:
- `/path/to/video` is the location of the video to use in the algorithm;
- `ALGORITHM` is a string used to specify which kind of algorithm should be run on the declared video. In particular, the following are available:
    - `lk` (Lucas-Kanade), i.e use Lucas-Kanade optical flow to track salient points;
    - `bfm` (Bruteforce Matcher), i.e. use a Bruteforce Matcher to track keypoints and descriptors;
    - `mtm` (Multiple Template Matching), i.e. match a given template multiple times across the frame;
    - `k` (Kalman Filter), i.e. predict the next position of some salient points in the scene.

Moreover, the following optional arguments can be specified:
* `-h`, `--help`, show this help message and exit
* `--max-frames MAX_FRAMES`, `-mf MAX_FRAMES`, Max number of frame to analyse in the video [default: 1000]
* `--scale SCALE`, `-s SCALE`, Size for rescaling the video [default: 0.2]
* `--sampling-rate SAMPLING_RATE`, `-sr SAMPLING_RATE`, Sampling rate for updating keypoints [default: 50]
* `--output VIDEO_NAME`, `-o VIDEO_NAME`, If specified, saves the video as 'VIDEO_NAME' using the provided format [default: None]

### Lucas-Kanade
To run Lucas-Kanade optical flow on the specified video simply type the following:

```console
$ python alghisi_simone_229355.py lk /path/to/video FEATURE_DETECTOR
```

where `FEATURE_DETECTOR` is the algorithm used for initialising Lucas-Kanade Optical Flow and extract features from the video. The following are available:

- `gftt`, i.e. (Good Features to Track);
- `sift`;
- `orb`;

#### Example
```console
$ python alghisi_simone_229355.py lk test/Contesto_industriale1.mp4 gftt
```

#### Additional information
For further details please type:

```console
$ python alghisi_simone_229355.py lk /path/to/video FEATURE_DETECTOR --help
```

### Brute-Force Matcher
To run the Brute-Force Matcher simply type the following:

```console
$ python alghisi_simone_229355.py bfm /path/to/video DESCRIPTOR_EXTRACTOR
```

where `DESCRIPTOR_EXTRACTOR` is the algorithm used for extracting the keypoints and the descriptors from the video. The following are available:

- `sift`;
- `orb`;

#### Example
```console
$ python alghisi_simone_229355.py bfm test/Contesto_industriale1.mp4 orb
```

#### Additional information
For further details please type:

```console
$ python alghisi_simone_229355.py bfm /path/to/video DESCRIPTOR_EXTRACTOR --help
```

### Multiple Template Matching
To run the Multiple Template Matching simply type the following:

```console
$ python alghisi_simone_229355.py mtm /path/to/video tm_preprocess
```

where `tm_preprocess` is an additional function where default templates and/or preprocessing operations on the video can be specified, such as:
* `-t TEMPLATE`, to specify the path to a template. If not set, the user is asked to select a ROI;
* `-s`, to save the template in the templates folder (useful when no template is specified and you want to save the selected ROI); 
* `-gb`, to apply Gaussian Blur to the video.
* ... 

#### Example
For example, 
```console
$ python alghisi_simone_229355.py mtm test/Contesto_industriale1.mp4 tm_preprocess -t templates/2022-05-18-09-36-21-817761.jpg -gb -c 150 200
```
runs the Multiple Template Matching by applying Gaussian Blur and the Canny edge-detector algorithms (specifying upper and lower bounds), considering the template `2022-05-18-09-36-21-817761.jpg` in the specified folder `templates`

#### Additional information
For further details please type:

```console
$ python alghisi_simone_229355.py mtm /path/to/video tm_preprocess --help
```

### Kalman Filter
To run the Kalman Filter on the specified video simply type the following:

```console
$ python alghisi_simone_229355.py k /path/to/video OBSERVATION_DETECTOR
```

where `OBSERVATION_DETECTOR` is the algorithm used for providing to the Kalman Filter the observations required for the matrices update. The following are available:

- `gftt`, i.e. (Good Features to Track);
- `sift`;
- `orb`;

#### Example
```console
$ python alghisi_simone_229355.py k test/Contesto_industriale1.mp4 orb
```

#### Additional information
For further details please type:

```console
$ python alghisi_simone_229355.py k /path/to/video OBSERVATION_DETECTOR --help
```