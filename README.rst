#########################################
DAY AND NIGHT IMAGE ANALYSIS & PREDICTION
#########################################

OVERVIEW
========

`Exploring Features <https://github.com/ayivima/day_night_image_analysis/blob/master/feature_engineering/feature_exploration.md/>`_

`Modelling and Prediction Overview <https://github.com/ayivima/day_night_image_analysis/blob/master/model_notebook/modelling_and_prediction.md/>`_


HIGHLIGHTS
==========

Box Plots for Feature Distribution of Training Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: feature_engineering/output_11_0.png


Image Clusters and Separation based on features (Find out more about features `here <https://github.com/ayivima/day_night_image_analysis/blob/master/feature_engineering/feature_exploration.md/>`_)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Contrast and Supraluminance

.. image:: feature_engineering/output_22_0.png

Contrast and Luminance

.. image:: feature_engineering/output_16_0.png


Sample Predictions (Find out more about modelling and predictions `here <https://github.com/ayivima/day_night_image_analysis/blob/master/model_notebook/modelling_and_prediction.md/>`_)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   predict("images/test3.jpg")

MODEL PREDICTIONS:
NearestCentroid->night, LogisticRegression->night, K Nearest Neighbors->night

.. image:: model_notebook/output_21_1.png



.. code-block:: python

   predict("images/test12.jpg")

MODEL PREDICTIONS:
NearestCentroid->day, LogisticRegression->day, K Nearest Neighbors->day

.. image:: model_notebook/output_30_1.png


.. code-block:: python

   predict("images/test10.jpg")

MODEL PREDICTIONS:
NearestCentroid->night, LogisticRegression->day, K Nearest Neighbors->night

.. image:: model_notebook/output_28_1.png



