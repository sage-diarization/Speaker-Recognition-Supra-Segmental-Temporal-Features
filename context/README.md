# Speaker-Recognition-Supra-Segmental-Temporal-Features

## Preface
This code has been highly optimized to our lab-infrastructure and thus most probably has to be modified.
The following parts are probably the most noteworthy regarding this:
- Data Generator
- Evaluation suite
- Setup

Since audios vary in shape, it gets difficult rather quickly to store them efficiently in a hierarchical data format like hdf5, zarr, etc.
Furthermore, we also draw frequency vectors from arbitrary locations in an audio for our study.
Those two factors made storing the audios in a "beautiful" way almost impossible.
We had to perform those modifications as our storage at the time was a fileserver which would've slowed training and testing down by factor 100+.

## Setup

```bash
cd src/
uv venv --python=3.8
source .venv/bin/activate
pip install -r requirements.txt
```
## Training

```bash
python train.py
```