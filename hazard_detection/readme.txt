This folder contains the complete, automated pipeline for training and evaluating a YOLOv8n hazard detection model. The pipeline is contained in a single script designed for Google Colab.


The script automates all steps of the machine learning lifecycle:

1.  Environment Setup
2.  Data Ingestion
3.  Model Training
4.  Model Quantization
5.  Model Evaluation


  * Automated Environment Setup: Installs `ultralytics` and `pandas`.
  * Data Management: Mounts Google Drive, finds `hazard.zip`, copies it to the Colab runtime, and unzips it.
  * 3-Stage Training: Implements a resumable, 3-stage training process for optimal accuracy.
  * Model Quantization: Automatically exports the final PyTorch model to TFLite FP16.
  * Comparative Analysis: Runs a CPU-based validation on both the PyTorch and TFLite models.
  * Report Generation: Prints a detailed comparison table of accuracy and performance metrics.


The script does not include the dataset. It relies on a specific setup in the user's Google Drive.

1.  A `hazard.zip` file must be placed in the root of the user's Google Drive (`MyDrive`).

2.  This `zip` file must contain the dataset and the `data.yaml` file.

3.  The expected folder structure inside the zip file is:

    ```
    ultimate_hazard_dataset/
    └── yolo/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
    ```

4.  The `data.yaml` file must be pre-configured with the correct paths (relative to its location) and class names.


The pipeline runs a 3-stage training process. All results, logs, and model weights are saved to `MyDrive/Security-Rover-Final/`.

  * Stage 1: Backbone training at 640px for 80 epochs with a learning rate of 0.005.
  * Stage 2: Model adaptation at 320px for 30 epochs with a learning rate of 0.001.
  * Stage 3: Fine-tuning at 320px for 20 epochs with a learning rate of 0.0002.

After training, the script finds the best weights from Stage 3 (`best.pt`) and exports them to a TFLite FP16 model.

The final quantized model is saved to:
`MyDrive/Security-Rover-Final/stage4_final_output/best_native_float16.tflite`


Finally, the script loads both the final PyTorch model (`best.pt`) and the TFLite FP16 model. It runs validation on the "test" split for both models.

The evaluation is forced to run on the CPU to provide a realistic performance baseline for edge devices.

A final report is printed to the console, comparing the following metrics for both models:

  * Class
  * Metric (Precision, Recall, mAP50, mAP50-95)
  * PyTorch FP32
  * TFLite FP16
  * Delta
  * Speed (Latency, FPS)
  * Model Size (MB)


1.  Place your prepared `hazard.zip` file in the root of your Google Drive.
2.  Open a new Google Colab notebook.
3.  Copy the entire content of `colab_train.py` into a single Colab cell.
4.  Run the cell. The entire pipeline will execute from start to finish.
