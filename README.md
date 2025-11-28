# PCB Defect Detection
## Design Assumptions:

The project will utilize ready-made datasets containing graphics of typical PCB defects (Justify the dataset selection).  

Prepare, augment, and normalize data for the model.  

Model selection and preparation, justification of the decision, presentation of comparisons between models. Metrics and other desired measured values.  

Preparation of a GUI application for presenting results.  

Preparation of a presentation describing the process.  

## Dataset Selection:

[Dataset Link - option 1](https://www.kaggle.com/datasets/norbertelter/pcb-defect-dataset/data)
[Dataset Link - option 2](https://www.kaggle.com/datasets/akhatova/pcb-defects)

Why this datasets?  
It contains 10668 images and the corresponding annotation files, which is perfect for this project.
Classified defects:
- missing hole,
- mouse bite,
- open circuit,
- short, 
- spur,
- spurious copper.

The data has enough images to train our model to detect defects, and if we are not focused on the percentage maxing it should be good choice. The quality of dataset is great and the resolution satisfies our needs.  

