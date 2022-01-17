# Database

### Copy database
    rsync -r -e "ssh -p 3022" anzem@dingo.fri.uni-lj.si:/hdd2/vol1/deepfakeDatabases/original_videos/Celeb-DF-v2 .
  
  or

    scp -r -P 3022 anzem@dingo.fri.uni-lj.si: /hdd2/vol1/deepfakeDatabases/original_videos/Celeb-DF-v2 .

### List of databases:
* UADFV
* DeepfakeTIMIT
* DFDGoogle
* FaceForensics++
* Celeb-DF-v2

### Databases location on Dingo:
/hdd2/vol1/deepfakeDatabases/original_videos/

### DB directory structure
    ├── Celeb-DF-v2
    │   ├── Celeb-real
    │   ├── Celeb-synthesis
    │   └── YouTube-real
    ├── DeepfakeTIMIT
    │   ├── dftimitreal-frames
    │   ├── higher_quality
    │   ├── lower_quality
    │   └── vidtimitreal
    ├── DFDGoogle
    │   ├── manipulated_sequences
    │   │   └── DeepFakeDetection
    │   └── original_sequences
    │       └── actors
    ├── FaceForensics++
    │   ├── manipulated_sequences
    │   │   ├── Deepfakes
    │   │   ├── Face2Face
    │   │   ├── FaceShifter
    │   │   ├── FaceSwap
    │   │   └── NeuralTextures
    │   └── original_sequences
    │       └── youtube
    └── UADFV
        ├── fake
        └── real
