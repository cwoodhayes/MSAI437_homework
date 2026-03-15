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

![Attention Heatmaps for GOOD examples](Attention%20Grid%20for%20GOOD%20Examples_grid.png)
> **Good Examples (above)**: Attention heatmaps for 10 input examples in which the correct answer token was predicted with >80% proability, as well as an average heatmap across all 10.

![Attention Heatmaps for BAD examples](Attention%20Grid%20for%20BAD%20Examples_grid.png)
> **Bad examples (above)**: Attention heatmaps for 10 input examples in which the correct answer token was predicted with <5% proability, as well as an average heatmap across all 10.

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

Overall, the heatmaps look surprisingly similar. I was expecting the attention maps to be substantially different between the two, and for the bad results to show extremely disordered attention across the board. However, there is only one obvious difference that points to a specific pathology in (at least most) bad results:

Good results focus attention primarily on input `b`, while bad results focus attention on other operands or diffusely across some set of them without particular focus on `b` (or, in the case of Example 5, on no operands at all). Other operands are irrelevant to the `b * b` problem, so they should be ignored. 
Interestingly, although all successful answers focus mostly on `b` of the 4, many include additional focus on a 2nd operand. This makes sense, as most of the training data contains binary operations on 2 different operands rather than the same one twice.

Also of note is that the no-peek mask is clearly visible as an upper right triangle in the results, indicating it's been implemented correctly across good & bad examples.


### e. Document any modifications you made to the code to produce your attention heatmaps.

- Added capture of attention tensors inside `attention()` using globals (`last_scores`, `last_output`) and a rolling buffer (`last_10_scores`) for multi-example averaging.
- Added `plot_attention_grid()` to show heatmaps of a set of inference runs, as well as an attention heatmap averaged across all given examples.
- Integrated these plotting calls in `example()` to render both a single-example heatmap for the last input, and an averaged heatmap for all 10 inputs (for both good and bad). Also refactored some code into helper functions for cleanliness.

## Part 3
First i tried to implement a multiplicative gain on just the b value's index (4) applied to the input to the softmax function, to give a higher attention weight to the value of b from all tokens. This helped for some examples but not others; I figured out that this is due to softmax inputs just as likely being negative as positive, so for negative values it actually pulled the outputs down.
I then switched to an additive augmentation rather than multiplicative gain; this ran into the problem that the augmentation was inconsistently applied depending upon how the other tokens were weighted; in some cases, this augmentation dominated, making the network forget to pay attention to other important tokens, while in others it was a drop in the bucket compared to other large-valued tokens, showing no change.

Finally, I realized that what really mattered was that we _ignore_ the other operand tokens; attention to b could mostly work itself out if the others were ignored.

This is the final solution I ran with -- subtracting `1e9` from the other argument tokens, and adding `1` to the `b` argument token, resulting in an average score of 81%, for a 78% improvment on this test set.

I implemented this using 2 global variables, `focus_indices` and `focus_augs` applied prior to softmax in the `attention()` function, and which i set to the values above prior to re-running on the bad dataset.

Obviously, this approach is tailored specifically to this particular input operation `b*b`, and will break all the other cases. We could fix that by automatically applying the augmentation to the correct input arguments at runtime, but this would defeat the whole point of a neural network; it's hardcoding. Really this needs to be solved with training tricks in any real scenario.

However, it's an interesting think to observe and mess around with in a pet environment like this.

![Attention Heatmaps for IMPROVED examples](Attention%20Grid%20for%20IMPROVED%20Examples_grid.png)
> **Improved examples (above)**: Attention heatmaps for the 10 bad input examples in which the correct answer token was originally predicted with <5% proability, with the augmentation described above applied, resulting in a 79% average score improvement.