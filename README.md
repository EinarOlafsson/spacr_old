# SpaCr
**S**patial **p**henotype **a**nalysis of **cr**isp screens (SpaCr). A collection of functions for generating measurement and classification data from microscopy images from high content imaging screens. Each notebook (.ipynb files) comes with an associated (.py) file which is downloaded when the notebook is run in Jupyter. Each notebook requires Jupyter and anaconda to be installed on the system. For notebooks that use torch (classification, finetune_cellpose, data_generation) a CUDA 11.8 compatible GPU is required.

Notebooks:
data_generation - Features:
 - Batch normalization of high content image data.
 - Cellpose mask generation of up to 3 object classes.
 - Generate object level measurements (~800).
 - Generate object level images for data visualization and Deep learning classification.
 - Metadata, measurement, or manual single object image annotation.
 - visualize measurement data (alpha).
   
classification - Features:
 - Use single object images (e.g. generated in data_generation) to train a torch model.
 - Apply trained models to image data.
 - Link classification data to sequencing data and perform regression analysis (alpha).

finetune_cellpose - Features:
 - Generate masks with an existing cellpose model.
 - Manually generate/modify object masks.
 - Fine-tune or train from scratch cellpose models (alpha).

simulate_screen - Features:
 -  simulate  the parameters of a CRISPR/Cas9 spatial phenotype screen (alpha).
