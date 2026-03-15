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
- I selected the first 10 "bad" observations where that same probability is `< 5%`.
- This is implemented in `select_good_and_bad_examples()`.

### b. Display the “good” and “bad” observations separately
Done; see CLI output from command above.

### c. Generate a heatmap of attention for both “good” and “bad” observations using the d_k = 128 model
Done; see CLI output from command above.

### d. Interpret these results to explain the observed differences in performance between “good” and “bad” observation subsets

Overall, the heatmaps look surprisingly similar. I was expecting the attention maps to be substantially different between the two, and for the bad results to show disordered attention (ie )

### e. Document any modifications you made to the code to produce your attention heatmaps.

- Added capture of attention tensors inside `attention()` using globals (`last_scores`, `last_output`) and a rolling buffer (`last_10_scores`) for multi-example averaging.
- Added `plot_attention_grid()` to show heatmaps of a set of inference runs, as well as an attention heatmap averaged across all given examples.
- Integrated these plotting calls in `example()` to render both a single-example heatmap for the last input, and an averaged heatmap for all 10 inputs (for both good and bad)

## Part 3