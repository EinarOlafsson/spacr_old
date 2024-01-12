# SpaCr
Spatial phenotype analysis of crisp screens (spacr). A collection of functions for generating measurement and classification data from microscopy images from high content imaging screens. Each notebook (.ipynb files) comes with an assoceated (.py) file which is downloaded when the notebook is run in Jupyter. Each notebook requires jupyter and anaconda to be installed on the system. For notebooks that use torch (classefication, finetune_cellpose, data_generation) a cuda 11.8 compatible GPU is required.

Notebooks:
data_generation - Features:
 - Batch notmalization of high content image data.
 - Cellpose mask generation of ut to 3 object classes.
 - Generate object level measurements (~800).
 - Generate object level images for data visualization and Deep learning classefication.
 - Metadata, measurement or manual single object image annotation.
 - visualize measurement data (alpha).
   
Classefication - Features:
 - Use single object images (e.g. generated in data_generation) to train a torch model.
 - Apply trained models to immage data.
 - Link classefication data to sequencing data and perform regression analasys (alpha).

finetune_cellpose - Features:
 - Generate masks with an existing cellpose model.
 - Manually generate/modify object masks.
 - Fine-tune or train from scratch cellpose models (appha).

simulate_screen - Features:
 -  simmulate the paramiters of a CRISPR/Cas9 spatial phenotype screen (alpha).
