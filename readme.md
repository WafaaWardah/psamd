# P.SAMD: Single-ended perceptual approaches for multi-dimensional analysis
This repository contains the implementation of the candidate model for this P.SAMD work item relevant to the Study Group 12 / Question 9, of the International Telecommunication Union (ITU-T).

## To set up

Install the dependencies (requirements.txt) and download the PXXX.py and the wrapper.py scripts. These are the only scripts needed for inference. The wrapper does not form an integral part of the recommendation, but is provided as an optional aid for users when preparing input speech files from unstructured or live scenarios.

### Download Weights
There are 5 weight  ```.pth``` files for each dimension. Download and save them in the same directory as the PXXX.py script. The weights can be downloaded [here](https://tubcloud.tu-berlin.de/s/K9owXP3Fnj4pnJg). 

## Model Usage

You can run prediction for either one .wav file or for all .wav files in a directory (or .pcm). You can also select if you want to only predict some dimensions to save compute time. 

### Command for assessing the quality of one .wav file

Assess the overall quality as well as all quality dimensions for a single .wav file and show the results on screen only (no output file will be created):

```bash
python run_predict.py --path /path/to/file.wav --print True
```

Assess selected dimensions (e.g., overall quality mos and noisiness) for a single .wav file and show the results on screen only (no output file will be created):

```bash
python run_predict.py --path /path/to/file.wav --dims mos noi --print True
```

Assess all dimensions for a single .wav file without printing the results, and save the predictions to an output file:

```bash
python run_predict.py --path /path/to/file.wav --output_dir /outputs/directory
```

### Command for assessing the quality of all .wav files in a directory

Assess all dimensions for all .wav files in a directory and display the predictions on screen only (no output files will be created):

```bash
python run_predict.py --path /path/to/directory --print True
```

Assess all dimensions for all .wav files in a directory using a specified batch size and number of workers, enable GPU processing, and save the predictions to an output file without printing them on screen:

```bash
python run_predict.py \
    --path /path/to/directory \
    --output_dir /outputs/directory \
    --bs 64 \
    --nw 4 \
    --device gpu
```

### Applying the pre-processing wrapper function for the input signal

Use the --prep argument to preprocess the input signal.

```bash
python run_predict.py \
    --path /path/to/directory \ 
    --print True \
    --prep True 
```

### Quick Conformance Check
The samples directory contains two sample speech files. This command can be used to predict their quality scores:

```bash
python run_predict.py --path samples --print True
```

Note that these results are obtained with the P.SAMD weights. The results should be:

```bash
(1/2) c00007_P501_C_english_m2_FB_48k.wav | MOS: 4.32, NOI: 4.58, DIS: 4.37, COL: 4.62, LOUD: 4.71
(2/2) c00001_P501_C_english_f1_FB_48k.wav | MOS: 4.43, NOI: 4.62, DIS: 4.27, COL: 4.71, LOUD: 4.69
```

The full official conormance test is provided in the Annex A of the Recommendation text.

# SQ-AST Model
The candidate model is a transformer-based speech quality prediction model. An initial version of the model was first published in [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) and can be cited as follows:

Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683

The first set of model weights from the Interspeech 2025 paper can be downloaded from [here](https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A). After the SQ-AST model was published in Interspeech 2025, the model was further trained on new datasets. This produced the improved weights that have been evaluated by it ITU-T SG12 Q9 and approved for standardization. Those weights are mentioned and linked earlier in this document. Note: the coloration weight file is the same, as there was no further improvement.

