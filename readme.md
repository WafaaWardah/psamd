# ITU-T Rec. P.566 Implementation
This is a personally maintained repository for the ITU-T Recommendation P.566 Single-ended machine-learning-based approaches for multi-dimensional analysis model. Previously, terms including P.SAMD or SQ-AST have also been used and may still be present in some documents. This repository may contain notes and updates that cannot be added to the official ITU-T attachments directly without due official processes in a short time.

The implementation is a transformer-based speech quality prediction model. It was first released as SQ-AST in 2025, and [this paper](https://www.isca-archive.org/interspeech_2025/wardah25_interspeech.html) can be cited when using this model, along with the Recommendation reference.

Cite as: Wardah, W., Spang, R.P., Barriac, V., Reimes, J., Llagostera, A., Berger, J., Möller, S. (2025) SQ-AST: A Transformer-Based Model for Speech Quality Prediction. Proc. Interspeech 2025, 2335-2339, doi: 10.21437/Interspeech.2025-2683

## To set up

Install the dependencies (requirements.txt) and download the P566.py script. This is the only script needed for inference. 

### Download Weights
The weight .pth files can be downloaded [here](https://tubcloud.tu-berlin.de/s/K9owXP3Fnj4pnJg). There are 5 of them for each dimension. Download and save them in the same directory as the P566.py script.

Previously published weights from the 2025 Interspeech publication can be downloaded from [here](https://tubcloud.tu-berlin.de/s/rik9dQaR66R8w5A). Note: the coloration weight file is the same, as there was no further improvement beyond this training.

## Running Predictions

You can run prediction for either one .wav file or for all .wav files in a directory. You can also select if you want to only predict some dimensions to save computation time. In addition to passing already-prepared 8-12 second-long speech files to the model, unprocessed multi-minute audio streams can be processed and assessed by the model (see end of README for instructions).

### Command for inference of one .wav file

Predict all dimensions for a single .wav file and show the results on screen only (no output file will be created):

```bash
python P566.py --path /path/to/file.wav --print True
```

Predict selected dimensions (e.g., overall quality mos and noisiness) for a single .wav file and show the results on screen only (no output file will be created):

```bash
python P566.py --path /path/to/file.wav --dims mos noi --print True
```

Predict all dimensions for a single .wav file without printing the results, and save the predictions to an output file:

```bash
python P566.py --path /path/to/file.wav --output_dir /outputs/directory
```

### Command for inference of all wav files in a directory

Predict all dimensions for all .wav files in a directory and display the predictions on screen only (no output files will be created):

```bash
python P566.py --path /path/to/directory --print True
```

Predict all dimensions for all .wav files in a directory using a specified batch size and number of workers, enable GPU processing, and save the predictions to an output file without printing them on screen:

```bash
python P566.py \
    --path /path/to/directory \
    --output_dir /outputs/directory \
    --bs 64 \
    --nw 4 \
    --device gpu
```

### Conformance Check
The databases are the same as those used for ITU-T P.863.2 conformance testing and can be accessed [here](https://www.itu.int/myworkspace/t-signals/vectors?val=100015010).

Additionally, a quick check can be done using the two sample speech files in the samples directory. This command can be used to predict their quality scores:

```bash
python P566.py --path samples --print True
```

Note that these results are from the improved weights (the first link). The results should be:

```bash
(1/2) c00007_P501_C_english_m2_FB_48k.wav | MOS: 4.32, NOI: 4.58, DIS: 4.37, COL: 4.62, LOUD: 4.71
(2/2) c00001_P501_C_english_f1_FB_48k.wav | MOS: 4.43, NOI: 4.62, DIS: 4.27, COL: 4.71, LOUD: 4.69
```

### Applying the recommended wrapper for preprocessing the input signal

Use the --prep argument to preprocess the input signal(s) before running P.SAMD.

```bash
python P566.py \
    --path /path/to/directory \ 
    --print True \
    --prep True 
```


