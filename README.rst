#########################################
DAY AND NIGHT IMAGE ANALYSIS & PREDICTION
#########################################

OVERVIEW
========
Find out more:

`Exploring Features <https://github.com/ayivima/day_night_image_analysis/blob/master/feature_engineering/feature_exploration.md/>`_

`Modelling and Prediction Overview <https://github.com/ayivima/day_night_image_analysis/blob/master/model_notebook/modelling_and_prediction.md/>`_


HIGHLIGHTS
==========

Box Plots for Feature Distribution of Training Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://raw.githubusercontent.com/ayivima/day_night_image_analysis/master/feature_engineering/output_11_0.png?token=ALUCYRU6ZHUEN4ET4OYSDTS5V3PBG


Image Clusters and Separation based on Contrast and Supraluminance. (Find out more about features `here <https://github.com/ayivima/day_night_image_analysis/blob/master/feature_engineering/feature_exploration.md/>`_)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. image:: https://raw.githubusercontent.com/ayivima/day_night_image_analysis/master/feature_engineering/output_22_0.png?token=ALUCYRW7DY7XYOS7TCDIUJK5V3QFE


Sample Predictions
^^^^^^^^^^^^^^^^^^

.. code-block:: python

  >> predict("images/test3.jpg")

MODEL PREDICTIONS:
NearestCentroid->night, LogisticRegression->night, K Nearest Neighbors->night

.. image:: https://raw.githubusercontent.com/ayivima/day_night_image_analysis/master/model_notebook/output_21_1.png?token=ALUCYRQOGWTJQO46EMTLONC5V3RTI


.. code-block:: python

  >> predict("images/test12.jpg")

MODEL PREDICTIONS:
NearestCentroid->day, LogisticRegression->day, K Nearest Neighbors->day

.. image:: https://raw.githubusercontent.com/ayivima/day_night_image_analysis/master/model_notebook/output_30_1.png?token=ALUCYRTE755GBJVVZJ7GYG25V3SUO
