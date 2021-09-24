.. IMPACTS documentation master file, created by
   sphinx-quickstart on Wed Jun  2 18:18:46 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

The Investigation of Microphysics and Precipitation for Atlantic Coast-Threatening Snowstorms (IMPACTS)
=======================================================================================================

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: documentation

   impacts_tools
   installation
   contributing


.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Example Notebooks

   notebooks/plot_radar_xsections.ipynb



.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Links

   GitHub Repository <https://github.com/torimcd/impacts_tools>
   IMPACTS Data Archive <https://ghrc.nsstc.nasa.gov/home/micro-articles/investigation-microphysics-and-precipitation-atlantic-coast-threatening-snowstorms>
   IMPACTS Field Catalog <http://catalog.eol.ucar.edu/impacts_2020>
   NASA ESPO IMPACTS Home Page <https://espo.nasa.gov/impacts/content/IMPACTS>

The Investigation of Microphysics and Precipitation for Atlantic Coast-Threatening Snowstorms (IMPACTS) 
is a NASA Earth Venture Suborbital-3 (EVS-3) aircraft-based field campaign to study precipitation banding
in severe winter cyclones in the Northeast United States.


What is IMPACTS-TOOLS?
======================
This package provides an object-oriented data model for observations collected during the Investigation of Microphysics 
and Precipitation for Atlantic Coast-Threatening Snowstorms (IMPACTS) Field Campaign. 
By wrapping the datasets from the various instruments, exposing their fields through a common interface, and including 
some light processing functions, the impacts-tools package makes it faster and easier to understand and analyze data 
collected during IMPACTS. 

IMPACTS data consists of various radars, microphysics probes, and other instruments flown on two aircraft
during snowstorms in winter 2020, 2022, and 2023.  IMPACTS_TOOLS may eventually contain more specialized functions 
for advanced analysis as papers are published by the IMPACTS Science Team, but the initial goals are to make things 
that should be easy - reading the data files, filtering by time and flight leg, plotting data from multiple instruments on 
the same figures - easier for a non-expert to get started with. 


Links
=====

* `GitHub Repository <https://github.com/torimcd/impacts_tools>`_
* `IMPACTS Data: GHRC DAAC <https://ghrc.nsstc.nasa.gov/home/micro-articles/investigation-microphysics-and-precipitation-atlantic-coast-threatening-snowstorms>`_
* `IMPACTS Field Catalog <http://catalog.eol.ucar.edu/impacts_2020>`_
* `NASA ESPO IMPACTS Home Page <https://espo.nasa.gov/impacts/content/IMPACTS>`_

* :ref:`search`
