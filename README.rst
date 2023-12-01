`cu_cat`
===========

`cu_cat` is an end-to-end gpu Python library that encodes categorical variables into machine-learnable numerics.
It is a cuda accelerated port of what was dirty_cat, now rebranded as `cu_cat <https://github.com/cu_cat-data/cu_cat>`_

What can `cu_cat` do?
------------------------

`cu_cat` provides tools (``TableVectorizer``...) and
encoders (``GapEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

`Example notebooks <https://github.com/graphistry/cu-cat/tree/master/examples/cu-cat_demo.ipynb>`_
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

Major dependencies the cuml and cudf libraries, as well as `standard python libraries <https://github.com/cu_cat-data/cu_cat/blob/main/setup.cfg>`_

Related projects
----------------

dirty_cat is now rebranded as part of the sklearn family as `cu_cat <https://github.com/cu_cat-data/cu_cat>`_

<<<<<<< HEAD
=======
Contributing
------------

The best way to support the development of skrub is to spread the word!

Also, if you already are a skrub user, we would love to hear about your use cases and challenges in the `Discussions <https://github.com/skrub-data/skrub/discussions>`_ section.

To report a bug or suggest enhancements, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ and/or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.

Additional resources
--------------------

* `Introductory video (YouTube) <https://youtu.be/_GNaaeEI2tg>`_
* `JupyterCon 2023 talk (YouTube) <https://youtu.be/lvDN0wgTpeI>`_
* `EuroSciPy 2023 poster (Dropbox) <https://www.dropbox.com/scl/fi/89tapbshxtw0kh5uzx8dc/Poster-Euroscipy-2023.pdf?rlkey=u4ycpiyftk7rzttrjll9qlrkx&dl=0>`_

References
----------

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
>>>>>>> master
