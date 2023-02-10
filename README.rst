`cuCat`
===========

.. image:: https://dirty-cat.github.io/stable/_static/cuCat.svg
   :align: center
   :alt: cuCat logo


|py_ver| |pypi_var| |pypi_dl| |codecov| |circleci| |black|

.. |py_ver| image:: https://img.shields.io/pypi/pyversions/cuCat
.. |pypi_var| image:: https://img.shields.io/pypi/v/cuCat?color=informational
.. |pypi_dl| image:: https://img.shields.io/pypi/dm/cuCat
.. |codecov| image:: https://img.shields.io/codecov/c/github/dirty-cat/cuCat/main
.. |circleci| image:: https://img.shields.io/circleci/build/github/dirty-cat/cuCat/main?label=CircleCI
.. |black| image:: https://img.shields.io/badge/code%20style-black-000000.svg

`cuCat <https://dirty-cat.github.io/>`_ is a Python library
that facilitates machine-learning on dirty categorical variables.

For a detailed description of the problem of encoding dirty categorical data, see
`Similarity encoding for learning with dirty categorical variables <https://hal.inria.fr/hal-01806175>`_ [1]_
and `Encoding high-cardinality string categorical variables <https://hal.inria.fr/hal-02171256v4>`_ [2]_.

If you like the package, please *spread the word*, and ⭐ `the repository <https://github.com/dirty-cat/cuCat/>`_!

What can `cuCat` do?
------------------------

`cuCat` provides tools (``TableVectorizer``, ``fuzzy_join``...) and
encoders (``GapEncoder``, ``MinHashEncoder``...) for **morphological similarities**,
for which we usually identify three common cases: **similarities, typos and variations**

`The first example notebook <https://dirty-cat.github.io/stable/auto_examples/01_cuCategories.html>`_
goes in-depth on how to identify and deal with dirty data using the `cuCat` library.

What `cuCat` does not
~~~~~~~~~~~~~~~~~~~~~~~~~

`Semantic similarities <https://en.wikipedia.org/wiki/Semantic_similarity>`_
are currently not supported.
For example, the similarity between *car* and *automobile* is outside the reach
of the methods implemented here.

This kind of problem is tackled by
`Natural Language Processing <https://en.wikipedia.org/wiki/Natural_language_processing>`_
methods.

`cuCat` can still help with handling typos and variations in this kind of setting.

Installation
------------

cuCat can be easily installed via `pip`::

    pip install cuCat

Dependencies
~~~~~~~~~~~~

Dependencies and minimal versions are listed in the `setup <https://github.com/dirty-cat/cuCat/blob/main/setup.cfg#L26>`_ file.

Related projects
----------------

Are listed on the `cuCat's website <https://dirty-cat.github.io/stable/#related-projects>`_

Contributing
------------

If you want to encourage development of `cuCat`,
the best thing to do is to *spread the word*!

If you encounter an issue while using `cuCat`, please
`open an issue <https://docs.github.com/en/issues/tracking-your-work-with-issues/creating-an-issue>`_ and/or
`submit a pull request <https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request>`_.
Don't hesitate, you're helping to make this project better for everyone!

Additional resources
--------------------

* `Introductory video (YouTube) <https://youtu.be/_GNaaeEI2tg>`_
* `Overview poster for EuroSciPy 2022 (Google Drive) <https://drive.google.com/file/d/1TtmJ3VjASy6rGlKe0txKacM-DdvJdIvB/view?usp=sharing>`_

References
----------

.. [1] Patricio Cerda, Gaël Varoquaux, Balázs Kégl. Similarity encoding for learning with dirty categorical variables. 2018. Machine Learning journal, Springer.
.. [2] Patricio Cerda, Gaël Varoquaux. Encoding high-cardinality string categorical variables. 2020. IEEE Transactions on Knowledge & Data Engineering.
