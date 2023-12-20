# Assembling: joining multiple tables {#assembling}

::: currentmodule
cu_cat
:::

Assembling is the process of collecting and joining together tables.
Good analytics requires including as much information as possible, often
from different sources.

\<\<\<\<\<\<\< HEAD cu_cat allows you to join tables on keys of
different types (string, numerical, datetime) with imprecise
correspondance. ======= skrub allows you to join tables on keys of
different types (string, numerical, datetime) with imprecise
correspondence. \>\>\>\>\>\>\> master

## Fuzzy joining tables

Joining two dataframes can be hard as the corresponding keys may be
different.

The `fuzzy_join`{.interpreted-text role="func"} uses similarities in
entries to join tables on one or more related columns. Furthermore, it
choose the type of fuzzy matching used based on the column type (string,
numerical or datetime). It also outputs a similarity score, to single
out bad matches, so that they can be dropped or replaced.

In sum, equivalent to `pandas.merge`{.interpreted-text role="func"}, the
`fuzzy_join`{.interpreted-text role="func"} has no need for
pre-cleaning.

## Joining external tables for machine learning

Joining is straigthforward for two tables because you only need to
identify the common key.

\<\<\<\<\<\<\< HEAD However, for more complex analysis, merging multiple
tables is necessary. cu_cat provides the `Joiner`{.interpreted-text
role="class"} as a convenient solution: multiple fuzzy joins can be
performed at the same time, given a set of input tables and key columns.
======= In addition, skrub also enable more advanced analysis:
\>\>\>\>\>\>\> master

-   `Joiner`{.interpreted-text role="class"}: fuzzy-join multiple
    external tables using a scikit-learn transformer, which can be used
    in a scikit-learn `~sklearn.pipeline.Pipeline`{.interpreted-text
    role="class"}. Pipelines are useful for cross-validation and
    hyper-parameter search, but also for model deployment.
-   `AggJoiner`{.interpreted-text role="class"}: instead of performing
    1:1 joins like Joiner, AggJoiner performs 1:N joins. It aggregate
    external tables first, then join them on the main table.
-   `AggTarget`{.interpreted-text role="class"}: in some settings, one
    can derive powerful features from the target [y]{.title-ref} itself.
    AggTarget aggregates the target without risking data leakage, then
    join the result back on the main table, similar to AggJoiner.

## Column selection inside a pipeline

Besides joins, another common operation on a dataframe is to select a
subset of its columns (also known as a projection). We sometimes need to
perform such a selection in the middle of a pipeline, for example if we
need a column for a join (with `Joiner`{.interpreted-text
role="class"}), but in a subsequent step we want to drop that column
before fitting an estimator.

skrub provides transformers to perform such an operation:

-   `SelectCols`{.interpreted-text role="class"} allows specifying the
    columns we want to keep.
-   Conversely `DropCols`{.interpreted-text role="class"} allows
    specifying the columns we want to discard.

## Going further: embeddings for better analytics

Data collection comes before joining, but is also an essential process
of table assembling. Although many datasets are available on the
internet, it is not always easy to find the right one for your analysis.

cu_cat has some very helpful methods that gives you easy access to
embeddings, or vectorial representations of an entity, of all common
entities from Wikipedia. You can use
`datasets.get_ken_embeddings`{.interpreted-text role="func"} to search
for the right embeddings and download them.

Other methods, such as
`datasets.fetch_world_bank_indicator`{.interpreted-text role="func"} to
fetch data of a World Bank indicator can also help you retrieve useful
data that will be joined to another table.
