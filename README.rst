`cu_cat`
===========

`cu_cat` is an end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics.
It is a cuda accelerated port of what was dirty_cat, now rebranded as `skrub <https://github.com/skrub-data/skrub>`_

What can `cu_cat` do?
------------------------

`cu_cat` provides tools (``TableVectorizer``...) and
encoders (``GapEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

`Example notebooks <https://github.com/graphistry/pygraphistry/tree/bio-demos/demos/demos_by_use_case/bio/gpu>`_
goes in-depth on how to identify and deal with dirty data (biological in this case) using the `cu_cat` library.

What `cu_cat` does not
~~~~~~~~~~~~~~~~~~~~~~~~~

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

`cu_cat` can still help with handling typos and variations in this kind of setting.

Installation
------------

cu_cat v 0.04 can be easily installed via `pip`::

    pip install git+http://github.com/graphistry/cu-cat.git@v0.04.0

Dependencies
~~~~~~~~~~~~

Major dependencies the cuml and cudf libraries, as well as `standard python libraries <https://github.com/skrub-data/skrub/blob/main/setup.cfg>`_

Related projects
----------------

dirty_cat is now rebranded as part of the sklearn family as `skrub <https://github.com/skrub-data/skrub>`_

