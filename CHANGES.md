---
---

::: currentmodule
cu_cat
:::

# Release 0.5.0

# Release 0.4.0 (beta 2)

## Major changes

\* [SuperVectorizer]{.title-ref} is renamed as
`TableVectorizer`{.interpreted-text role="class"}, a warning is raised
when using the old name. `484`{.interpreted-text role="pr"} by
`Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"} \*
New experimental feature: joining tables using
`fuzzy_join`{.interpreted-text role="func"} by approximate key matching.
Matches are based on string similarities and the nearest neighbors
matches are found for each category. `291`{.interpreted-text role="pr"}
by `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"}
and `Leo Grinsztajn <LeoGrin>`{.interpreted-text role="user"} \* New
experimental feature: `FeatureAugmenter`{.interpreted-text
role="class"}, a transformer that augments with
`fuzzy_join`{.interpreted-text role="func"} the number of features in a
main table by using information from auxilliary tables.
`409`{.interpreted-text role="pr"} by
`Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"} \*
**datasets.fetching**: contains a new function
`fetch_world_bank_indicator`{.interpreted-text role="func"} that can be
used to download any indicator from the World Bank Open Data platform,
the indicator ID that can be found there. `291`{.interpreted-text
role="pr"} by `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text
role="user"} \* Unnecessary API has been made private: everything
(files, functions, classes) starting with an underscore shouldn\'t be
imported in your code. `331`{.interpreted-text role="pr"} by
`Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"} \* The
`MinHashEncoder`{.interpreted-text role="class"} now supports a
[n_jobs]{.title-ref} parameter to parallelize the hashes computation.
`267`{.interpreted-text role="pr"} by
`Leo Grinsztajn <LeoGrin>`{.interpreted-text role="user"} and
`Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}. \* New
experimental feature: deduplicating misspelled categories using
`deduplicate`{.interpreted-text role="func"} by clustering string
distances. This function works best when there are significantly more
duplicates than underlying categories. `339`{.interpreted-text
role="pr"} by `Moritz Boos <mjboos>`{.interpreted-text role="user"}.

## Minor changes

-   Removed example [Fitting scalable, non-linear models on data with
    dirty categories]{.title-ref}. `386`{.interpreted-text role="pr"} by
    `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"}
-   `MinHashEncoder`{.interpreted-text role="class"}\'s
    `minhash`{.interpreted-text role="func"} method is no longer public.
    `379`{.interpreted-text role="pr"} by
    `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"}
-   Fetching functions now have an additional argument `directory`,
    which can be used to specify where to save and load from datasets.
    `432`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

## Bug fixes

-   The `MinHashEncoder`{.interpreted-text role="class"} now considers
    [None]{.title-ref} and empty strings as missing values, rather than
    raising an error. `378`{.interpreted-text role="pr"} by
    `Gael Varoquaux <GaelVaroquaux>`{.interpreted-text role="user"}

# Release 0.3.0

## Major changes

-   New encoder: `DatetimeEncoder`{.interpreted-text role="class"} can
    transform a datetime column into several numerical columns (year,
    month, day, hour, minute, second, \...). It is now the default
    transformer used in the `TableVectorizer`{.interpreted-text
    role="class"} for datetime columns. `239`{.interpreted-text
    role="pr"} by `Leo Grinsztajn <LeoGrin>`{.interpreted-text
    role="user"}

-   The `TableVectorizer`{.interpreted-text role="class"} has seen some
    major improvements and bug fixes:

    -   Fixes the automatic casting logic in `transform`.
    -   To avoid dimensionality explosion when a feature has two unique
        values, the default encoder
        (`~sklearn.preprocessing.OneHotEncoder`{.interpreted-text
        role="class"}) now drops one of the two vectors (see parameter
        [drop=\"if_binary\"]{.title-ref}).
    -   `fit_transform` and `transform` can now return unencoded
        features, like the
        `~sklearn.compose.ColumnTransformer`{.interpreted-text
        role="class"}\'s behavior. Previously, a `RuntimeError` was
        raised.

    `300`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   **Backward-incompatible change in the TableVectorizer**: To apply
    `remainder` to features (with the `*_transformer` parameters), the
    value `'remainder'` must be passed, instead of `None` in previous
    versions. `None` now indicates that we want to use the default
    transformer. `303`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   Support for Python 3.6 and 3.7 has been dropped. Python \>= 3.8 is
    now required. `289`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   Bumped minimum dependencies:

    -   scikit-learn\>=0.23
    -   scipy\>=1.4.0
    -   numpy\>=1.17.3
    -   pandas\>=1.2.0 `299`{.interpreted-text role="pr"} and
        `300`{.interpreted-text role="pr"} by
        `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   Dropped support for Jaro, Jaro-Winkler and Levenshtein distances.

    -   The `SimilarityEncoder`{.interpreted-text role="class"} now
        exclusively uses `ngram` for similarities, and the
        [similarity]{.title-ref} parameter is deprecated. It will be
        removed in 0.5. `282`{.interpreted-text role="pr"} by
        `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

## Notes

-   The `transformers_` attribute of the
    `TableVectorizer`{.interpreted-text role="class"} now contains
    column names instead of column indices for the \"remainder\"
    columns. `266`{.interpreted-text role="pr"} by
    `Leo Grinsztajn <LeoGrin>`{.interpreted-text role="user"}

# Release 0.2.2

## Bug fixes

-   Fixed a bug in the `TableVectorizer`{.interpreted-text role="class"}
    causing a `FutureWarning`{.interpreted-text role="class"} when using
    the `get_feature_names_out`{.interpreted-text role="func"} method.
    `262`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

# Release 0.2.1

## Major changes

-   Improvements to the `TableVectorizer`{.interpreted-text
    role="class"}

    > -   Type detection works better: handles dates, numerics columns
    >     encoded as strings, or numeric columns containing strings for
    >     missing values.

    `238`{.interpreted-text role="pr"} by
    `Leo Grinsztajn <LeoGrin>`{.interpreted-text role="user"}

-   `get_feature_names`{.interpreted-text role="func"} becomes
    `get_feature_names_out`{.interpreted-text role="func"}, following
    changes in the scikit-learn API.
    `get_feature_names`{.interpreted-text role="func"} is deprecated in
    scikit-learn \> 1.0. `241`{.interpreted-text role="pr"} by
    `Gael Varoquaux <GaelVaroquaux>`{.interpreted-text role="user"}

-   

    Improvements to the `MinHashEncoder`{.interpreted-text role="class"}

    :   -   It is now possible to fit multiple columns simultaneously
            with the `MinHashEncoder`{.interpreted-text role="class"}.
            Very useful when using for instance the
            `~sklearn.compose.make_column_transformer`{.interpreted-text
            role="func"} function, on multiple columns.

    `243`{.interpreted-text role="pr"} by
    `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"}

## Bug-fixes

-   Fixed a bug that resulted in the `GapEncoder`{.interpreted-text
    role="class"} ignoring the analyzer argument.
    `242`{.interpreted-text role="pr"} by
    `Jovan Stojanovic <jovan-stojanovic>`{.interpreted-text role="user"}
-   `GapEncoder`{.interpreted-text role="class"}\'s
    [get_feature_names_out]{.title-ref} now accepts all iterators, not
    just lists. `255`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}
-   Fixed `DeprecationWarning`{.interpreted-text role="class"} raised by
    the usage of [distutils.version.LooseVersion]{.title-ref}.
    `261`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

## Notes

-   Remove trailing imports in the `MinHashEncoder`{.interpreted-text
    role="class"}.
-   Fix typos and update links for website.
-   Documentation of the `TableVectorizer`{.interpreted-text
    role="class"} and the `SimilarityEncoder`{.interpreted-text
    role="class"} improved.

# Release 0.2.0

Also see pre-release 0.2.0a1 below for additional changes.

## Major changes

-   Bump minimum dependencies:

    -   scikit-learn (\>=0.21.0) `202`{.interpreted-text role="pr"} by
        `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}
    -   pandas (\>=1.1.5) **! NEW REQUIREMENT !**
        `155`{.interpreted-text role="pr"} by
        `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   **datasets.fetching** - backward-incompatible changes to the example
    datasets fetchers:

    -   The backend has changed: we now exclusively fetch the datasets
        from OpenML. End users should not see any difference regarding
        this.
    -   The frontend, however, changed a little: the fetching functions
        stay the same but their return values were modified in favor of
        a more Pythonic interface. Refer to the docstrings of functions
        [cu_cat.datasets.fetch\_\*]{.title-ref} for more information.
    -   The example notebooks were updated to reflect these changes.
        `155`{.interpreted-text role="pr"} by
        `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

-   **Backward incompatible change to**
    `MinHashEncoder`{.interpreted-text role="class"}: The
    `MinHashEncoder`{.interpreted-text role="class"} now only supports
    two dimensional inputs of shape (N_samples, 1).
    `185`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"} and
    `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text role="user"}.

-   Update [handle_missing]{.title-ref} parameters:

    -   `GapEncoder`{.interpreted-text role="class"}: the default value
        \"zero_impute\" becomes \"empty_impute\" (see doc).
    -   `MinHashEncoder`{.interpreted-text role="class"}: the default
        value \"\" becomes \"zero_impute\" (see doc).

    `210`{.interpreted-text role="pr"} by
    `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text role="user"}.

-   Add a method \"get_feature_names_out\" for the
    `GapEncoder`{.interpreted-text role="class"} and the
    `TableVectorizer`{.interpreted-text role="class"}, since
    [get_feature_names]{.title-ref} will be depreciated in scikit-learn
    1.2. `216`{.interpreted-text role="pr"} by
    `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text role="user"}

## Notes

-   Removed hard-coded CSV file
    [cu_cat/data/FiveThirtyEight_Midwest_Survey.csv]{.title-ref}.

-   Improvements to the `TableVectorizer`{.interpreted-text
    role="class"}

    -   Missing values are not systematically imputed anymore
    -   Type casting and per-column imputation are now learnt during
        fitting
    -   Several bugfixes

    `201`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}

# Release 0.2.0a1

Version 0.2.0a1 is a pre-release. To try it, you have to install it
manually using:

    pip install --pre cu_cat==0.2.0a1

or from the GitHub repository:

    pip install git+https://github.com/cu-cat/cu_cat.git

## Major changes

-   Bump minimum dependencies:
    -   Python (\>= 3.6)
    -   NumPy (\>= 1.16)
    -   SciPy (\>= 1.2)
    -   scikit-learn (\>= 0.20.0)
-   `TableVectorizer`{.interpreted-text role="class"}: Added automatic
    transform through the `TableVectorizer`{.interpreted-text
    role="class"} class. It transforms columns automatically based on
    their type. It provides a replacement for scikit-learn\'s
    `~sklearn.compose.ColumnTransformer`{.interpreted-text role="class"}
    simpler to use on heterogeneous pandas DataFrame.
    `167`{.interpreted-text role="pr"} by
    `Lilian Boulard <LilianBoulard>`{.interpreted-text role="user"}
-   **Backward incompatible change to** `GapEncoder`{.interpreted-text
    role="class"}: The `GapEncoder`{.interpreted-text role="class"} now
    only supports two-dimensional inputs of shape (n_samples,
    n_features). Internally, features are encoded by independent
    `GapEncoder`{.interpreted-text role="class"} models, and are then
    concatenated into a single matrix. `185`{.interpreted-text
    role="pr"} by `Lilian Boulard <LilianBoulard>`{.interpreted-text
    role="user"} and `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text
    role="user"}.

## Bug-fixes

-   Fix [get_feature_names]{.title-ref} for scikit-learn \> 0.21.
    `216`{.interpreted-text role="pr"} by
    `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text role="user"}

# Release 0.1.1

## Major changes

## Bug-fixes

-   RuntimeWarnings due to overflow in `GapEncoder`{.interpreted-text
    role="class"}. `161`{.interpreted-text role="pr"} by
    `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text role="user"}

# Release 0.1.0

## Major changes

-   `GapEncoder`{.interpreted-text role="class"}: Added online
    Gamma-Poisson factorization through the
    `GapEncoder`{.interpreted-text role="class"} class. This method
    discovers latent categories formed via combinations of substrings,
    and encodes string data as combinations of these categories. To be
    used if interpretability is important. `153`{.interpreted-text
    role="pr"} by `Alexis Cvetkov <alexis-cvetkov>`{.interpreted-text
    role="user"}

## Bug-fixes

-   Multiprocessing exception in notebook. `154`{.interpreted-text
    role="pr"} by `Lilian Boulard <LilianBoulard>`{.interpreted-text
    role="user"}

# Release 0.0.7

-   **MinHashEncoder**: Added `minhash_encoder.py` and `fast_hast.py`
    files that implement minhash encoding through the
    `MinHashEncoder`{.interpreted-text role="class"} class. This method
    allows for fast and scalable encoding of string categorical
    variables.
-   **datasets.fetch_employee_salaries**: change the origin of download
    for employee_salaries.
    -   The function now return a bunch with a dataframe under the field
        \"data\", and not the path to the csv file.
    -   The field \"description\" has been renamed to \"DESCR\".
-   **SimilarityEncoder**: Fixed a bug when using the Jaro-Winkler
    distance as a similarity metric. Our implementation now accurately
    reproduces the behaviour of the `python-Levenshtein` implementation.
-   **SimilarityEncoder**: Added a [handle_missing]{.title-ref}
    attribute to allow encoding with missing values.
-   **TargetEncoder**: Added a [handle_missing]{.title-ref} attribute to
    allow encoding with missing values.
-   **MinHashEncoder**: Added a [handle_missing]{.title-ref} attribute
    to allow encoding with missing values.

# Release 0.0.6

-   **SimilarityEncoder**: Accelerate `SimilarityEncoder.transform`, by:
    -   computing the vocabulary count vectors in `fit` instead of
        `transform`
    -   computing the similarities in parallel using `joblib`. This
        option can be turned on/off via the `n_jobs` attribute of the
        `SimilarityEncoder`{.interpreted-text role="class"}.
-   **SimilarityEncoder**: Fix a bug that was preventing a
    `SimilarityEncoder`{.interpreted-text role="class"} to be created
    when `categories` was a list.
-   **SimilarityEncoder**: Set the dtype passed to the ngram similarity
    to float32, which reduces memory consumption during encoding.

# Release 0.0.5

-   **SimilarityEncoder**: Change the default ngram range to (2, 4)
    which performs better empirically.
-   **SimilarityEncoder**: Added a [most_frequent]{.title-ref} strategy
    to define prototype categories for large-scale learning.
-   **SimilarityEncoder**: Added a [k-means]{.title-ref} strategy to
    define prototype categories for large-scale learning.
-   **SimilarityEncoder**: Added the possibility to use hashing ngrams
    for stateless fitting with the ngram similarity.
-   **SimilarityEncoder**: Performance improvements in the ngram
    similarity.
-   **SimilarityEncoder**: Expose a [get_feature_names]{.title-ref}
    method.
