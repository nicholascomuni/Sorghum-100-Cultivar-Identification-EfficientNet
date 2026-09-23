# Data

The data is **not included** in this repository. It belongs to the
[Sorghum-100 Cultivar Identification - FGVC 9](https://www.kaggle.com/competitions/sorghum-id-fgvc-9) competition
and is subject to the competition rules. Accept the rules on the competition page before downloading.

## What the notebook reads

The original run used `small-jpegs-fgvc`, a public Kaggle dataset uploaded by another participant that contains the
competition images re-encoded as smaller JPEG files (faster to read than the original PNGs). On Kaggle it is mounted
at `../input/small-jpegs-fgvc`, which is the default `DATA_DIR` in the notebook. The notebook expects this layout:

```text
<DATA_DIR>/
├── train_cultivar_mapping.csv   # columns: image, cultivar
├── train/                       # training images referenced by the "image" column
└── test/                        # test images (*.jpeg), flat folder
```

At submission time the test file names are converted from `.jpeg` back to the `.png` names used by the competition.

## Option A: Kaggle Notebook (recommended)

1. Create a new notebook and upload `KaggleSorghum100.ipynb` (File > Import Notebook).
2. Add data: the competition `sorghum-id-fgvc-9` and the dataset found by searching for `small-jpegs-fgvc`.
3. Enable a GPU accelerator and run all cells.

## Option B: local copy with the Kaggle CLI

```bash
pip install kaggle                     # needs ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)
kaggle competitions download -c sorghum-id-fgvc-9 -p data/
unzip -q data/sorghum-id-fgvc-9.zip -d data/sorghum-id-fgvc-9
```

The original competition archive ships full-resolution PNG images and its folder names may differ from the layout above.
Either download the `small-jpegs-fgvc` dataset from its Kaggle page (`kaggle datasets download -d <owner>/small-jpegs-fgvc`)
or arrange the original files into the layout above, then set `DATA_DIR` in the notebook accordingly
(for example `DATA_DIR = "data/small-jpegs-fgvc"`). If you use PNG test images, drop the `.jpeg` -> `.png`
conversion in the last cells.
