# advml

## Possible Dataset Choices

1. [Last FM](https://grouplens.org/datasets/hetrec-2011/)
    - with friends information
    - but no user's personal information
    - We could use this dataset to explore the question: how to utilize user social relations without content
2. [KKBox] https://www.kaggle.com/c/kkbox-music-recommendation-challenge/data
    - with user information
3. [MovieLens 20M Dataset](https://www.kaggle.com/grouplens/movielens-20m-dataset/data)
    - with two kinds of interaction (rating & tag)
    - can explore how to merge two kinds of interactions
4. still use the Recsys-17 dataset

## Related Resources

- [CDL keras implementation](https://github.com/zoujun123/Keras-CDL/blob/master/CDL.py)

## ideas

1. use word2vec for preference training
2. train a full connected layer from columns/rows to latent preference vector instead of WMF (which means)
3. just id + embedding (similar as CDL)

- in DropoutNet, wmf is used instead of mf because that the rating is kind of implicit

## Variations

1. Original DropoutNet
2. replace the underlying WMF
    - user/item id -> Embedding Layers -> Preference Latent vector;
    - output: modified rating (5, 20, -10, etc)
    - just use the known pairs
3. v2 + modification:
    - output: use $p = 1$ or $0$ instead of modified rating $r$
4. v3 + modification:
    - loss: add multiplier as $1 + \alpha r$
5. user preference history (the row in the preference matrix) -> Embedding layers (in this way, we don't have to retrain the model every time the preference table changes)
    - cannot directly use the rows or columns as input of dense: how to deal with null?
    - maybe consider $\frac{1}{\text{number of interactions}}\sum_i{v^{tag}_i}$?
        - maybe train two models: for 0 and for 1
        - may use tfidf to normalize
        

## todo list

1. re-implement
2. 实验embedding
3. 实验variation of embedding

## Working List

- first pass: modify all the tensorflow code to keras
- second pass: review and replace all the tensorflow related variables


## Data Overview

- `num_user = 30755`
- `num_item = 359966`


## Interface

- Feature: from raw to input data for main model
    - id mapping: all the id's used outside the folder is mapped_id
        - user_id_map: `[before_map],[mapped_id]`
        - song_id_map: `[before_map],[mapped_id]`
    - context
        - user_context, in csv format with headers
        - song_context, in csv format with headers
        - event_context, in csv format with headers
    - split (interaction), in csv format with headers:
        - all_interaction: `[user_id],[song_id],[target(0/1)]`
            - All the files below are made from subsets of rows in this table
        - train
        - test_warm, validation interactions for warm start
        - test_cold_item: validation interactions for item cold start
        - test_cold_user: validation interactions for user cold start
        - test_cold_item_item_ids: targets item ids for item cold start
        - test_cold_user_item_ids: target user ids for user cold start
- Model notebook
    - read in the data from feature
    - TODO train the MF model
    - DropoutNet flow: training
        - generate batch
            - firstly, sample user_batch_size=1000 of users
            - then, use the MF model
                - get topN items and scores
                - get randomN items and scores
                - merge them into one, called target
            - split the target into data batch
            - feed in network
                - 1st latent
                    - embed user_id, song_id
                - content
                    - embed categorical features
                    - concatenate with numerical features
                    - pass to dense layers
                - concatenate 1st latent and content
                - feed in dense layers and get 2nd latent
                - dot product the two 2nd latents, get the score
        - input
            - read in context
            - read in MF latent vector
    - evaluate: recall at k
        - prepare the all the items for user, called target list
        - get the ranking of all items, removing the ones from training
        - label: target list minus the ones from training
        - recall at k
            - notice there may be users with no positive feedback (not sure)


### Clean and split the data

1. remove the user_id and song_id that doesn't has any interaction entry
2. split the data into test_cold/test_warm: around 7,377,418 (7M) interactions
    - test size overall should be around 1.5M (around 20%)
    - test warm (0.5M=500K), sample 500K that:
        - the user has $\ge t_u=200$ interactions in total
        - the item has $\ge t_i=20$ interactions in total
    - test cold item (0.5M=500K), fetch all the remaining that:
        - the user has $\ge t_u=200$ interactions in total
        - the item has $< t_i=10$ interactions in total
        - around 580K
    - test cold user (0.5M=500K), fetch all the remaining that:
        - the user has $< t_u=120$ interactions in total
        - the item has $\ge t_i=10$ interactions in total
        - around 520K
    - train: all the remaining
    
## Experiment List

- exp1: normal
- exp2: try to be fast, but with bug
- exp3: RMSprop(1e-2)
    - the loss (0.10+) seems to be larger than 1e-3 (loss=0.06+)
    - so i decide to use 1e-5 to see the result
- exp4: choose (100,80) for hidden layers and 64 for latent dimension