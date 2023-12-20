#!/bin/bash -x

<<<<<<< HEAD
python -m pytest --pyargs cu_cat --cov=cu_cat
=======
pytest --pyargs skrub --cov=skrub -n auto --doctest-modules --cov-report xml
>>>>>>> master
