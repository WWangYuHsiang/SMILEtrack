# SMILEtrack
This code is based on the implementation of [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort)
# 1. Installation

SMILEtrack code is based on [ByteTrack](https://github.com/ifzhang/ByteTrack) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort)

Visit their installation guides for more setup options.

# 2. Download
PRB weight [link](https://drive.google.com/file/d/1BBkBTbdiUgwgeQ-BWbe1i1BbYXseHU4S/view?usp=share_link)

SLM weight [link](https://drive.google.com/file/d/1RDuVo7jYBkyBR4ngnBaVQUtHL8nAaGaL/view?usp=share_link)

# 3.Data Preparation
Download [MOT17](https://motchallenge.net/data/MOT17/) from the [official website](https://motchallenge.net/). And put them in the following structure:
```
<dataets_dir>
      │
      ├── MOT17
      │      ├── train
      │      └── test    
      
    
```
# 4.Tracking

By submitting the txt files produced in this part to MOTChallenge website and you can get the same results as in the paper.
Tuning the tracking parameters carefully could lead to higher performance. In the paper we apply ByteTrack's calibration.

## Track by detector YOLOX
```
cd <BoT-SORT_dir>
$ python3 tools/track.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "test" --fp16 --fuse
$ python3 tools/interpolation.py --txt_path <path_to_track_result>
```
## Track by detector PRBNet
```
cd <BoT-SORT_dir>
$ python3 tools/track_prb.py <dataets_dir/MOT17> --default-parameters --with-reid --benchmark "MOT17" --eval "test" --fp16 --fuse
$ python3 tools/interpolation.py --txt_path <path_to_track_result>
```
# 5.Tracking performance
## Results on MOT17 challenge test set
| Tracker        | MOTA           | IDF1  | HOTA |
| ------------- |:-------------:| -----:  | -----:|
| SMILEtrack       | 81.06      | 80.5 | 65.24 |

# 6.Acknowledgement
A large part of the codes, ideas and results are borrowed from [ByteTrack](https://github.com/ifzhang/ByteTrack), [BoT-SORT](https://github.com/NirAharon/BoT-SORT#bot-sort), thanks for their excellent work!

