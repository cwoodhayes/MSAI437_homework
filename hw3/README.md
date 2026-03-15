# MSAI 437 Homework 3

## Part 1
To generate the files requested, run this command:
```bash
python mathformer.py -d_model 128 -seqlen 20 -loadname weights_128 -n_layers 1 -heads 1 -run_single -idx 1979
```

## Part 2

To generate the plots shown below, run this command:
```bash
python mathformer.py -d_model 128 -seqlen 20 -loadname weights_128 -n_layers 1 -heads 1
```

### a. Describe your selection criteria for both the "good" and "bad" observations,

- I filtered `results.txt` to only expressions with operation `[EQ] ans = b * b`.
- From those, I selected the first 10 "good" observations where the reported probability of the correct answer (last token in each line in `results.txt`) is `> 80%`.
- I selected the first 10 "bad" observations where that same probability is `< 1%`.
- This is implemented in `select_good_and_bad_examples()`.

### e. Document any modifications you made to the code to produce your attention heatmaps.

- Added capture of attention tensors inside `attention()` using globals (`last_scores`, `last_output`) and a rolling buffer (`last_10_scores`) for multi-example averaging.
- Added `plot_attention()` to show an attention heatmap of a single example inference run.
- Added `plot_attention_multiple()` to show an attention heatmap averaged across multiple examples, which is intended to let me get a sense of shared performance without looking at 10 separate plots.  
- Integrated these plotting calls in `example()` to render both a single-example heatmap for the last input, and an averaged heatmap for all 10 inputs (for both good and bad)
