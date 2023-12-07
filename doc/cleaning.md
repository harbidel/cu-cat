# Cleaning

::: currentmodule
cu_cat
:::

## `deduplicate`{.interpreted-text role="func"}: merging variants of the same entry

`deduplicate`{.interpreted-text role="func"} is used to merge multiple
variants of the same entry into one, for instance typos. Such cleaning
is needed to apply subsequent dataframe operations that need exact
correspondences, such as counting elements. It is typically not needed
when the entries are fed directly to a machine-learning model, as a
`dirty-category encoder <dirty_categories>`{.interpreted-text
role="ref"} can suffice.
