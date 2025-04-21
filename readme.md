# AI_Beauty_Bias

**Case Study of AI Beauty Bias**  
Author: Chandon Hamel  
Regis University, MSDS 640

---

This repository contains the code for a study investigating ethnic bias in AI-powered facial beauty prediction models. The project evaluates how deep learning models trained on popular beauty datasets may encode and amplify societal biases, with a focus on fairness and ethical implications in real-world applications.

---

## Project Overview

**Objective:**  
To analyze and quantify ethnic bias in facial beauty assessment models using two benchmark datasets (SCUT-FBP5500 and MEBeauty) and to assess model generalization and fairness on a demographically balanced dataset (FairFace).

**Key Contributions:**
- Implementation of a reproducible pipeline for training, evaluating, and bias-testing beauty regressors.
- Statistical analysis of distributional and error parity across demographic groups.
- Ethical discussion and recommendations for mitigating bias in AI beauty systems.

---

## Repository Structure

| File                         | Description                                                                 |
|------------------------------|-----------------------------------------------------------------------------|
| `crop_face.py`               | Script for face detection and cropping using MTCNN.                         |
| `l_module.py`                | LightningModule for PyTorch Lightning model training.                       |
| `train_SCUT.py`              | Training script for SCUT-FBP5500 dataset.                                   |
| `train_MEB.py`               | Training script for MEBeauty dataset.                                       |
| `prepare_SCUT_csv.ipynb`     | Preprocessing notebook for SCUT-FBP5500 metadata.                           |
| `prepare_MEB_csv.ipynb`      | Preprocessing notebook for MEBeauty metadata.                               |
| `prepare_FairFace_csv.ipynb` | Preprocessing notebook for FairFace metadata.                               |
| `SCUT_analysis.ipynb`        | Analysis notebook for SCUT-FBP5500 results and bias metrics.                |
| `MEB_analysis.ipynb`         | Analysis notebook for MEBeauty results and bias metrics.                    |
| `FairFace_analysis.ipynb`    | Analysis notebook for FairFace cross-dataset evaluation.                    |
| `*.csv`, `*.txt`             | Metadata, labels, and prediction results for each dataset.                  |
| `.gitignore`, `.gitattributes` | Standard git configuration files.                                        |
| `requirements.txt`           | Python package dependencies.                                                |
| `README.md`                  | This README file.                                                          |
| `BeautyBias_Hamel.pdf`       | Full paper detailing the study and findings.                               |

---

## Datasets

- **SCUT-FBP5500**: 5,500 images, Asian and Caucasian subjects, beauty scores [1–5].
- **MEBeauty**: 2,550 images, six ethnic groups, beauty scores [1–10].
- **FairFace**: 86,744 images, seven race groups, used for cross-dataset evaluation and bias analysis.

*Note: Original images are not included.*

---

## Getting Started

**Requirements:**
- Python 3.8+
- PyTorch
- PyTorch Lightning
- Facenet-Pytorch
- OpenCV
- pandas, numpy, scikit-learn, scipy, matplotlib, seaborn

**Installation:**

`pip install -r requirements.txt`

**Preprocessing:**
- Use the `prepare_*_csv.ipynb` notebooks to generate metadata and label files for each dataset.
- Run `crop_face.py` to detect and crop faces in the image datasets.

**Training:**
- Train models on each dataset with `train_SCUT.py` and `train_MEB.py`.
- Model checkpoints and logs will be saved for subsequent analysis.

**Analysis:**
- Use the provided Jupyter notebooks (`*_analysis.ipynb`) to reproduce the statistical bias analysis and visualizations described in the paper.

---

## Reproducibility

- All code is modular and organized for end-to-end reproducibility.
- Statistical tests (Mann-Whitney U, Kruskal-Wallis, Dunn’s post hoc) are implemented in the analysis notebooks.
- Prediction results and error metrics are stored in CSV files for transparency.

---

## Results Summary

- Both SCUT- and MEBeauty-trained models exhibit significant ethnic bias in beauty score predictions, as measured by statistical parity and error parity metrics.
- Cross-dataset and FairFace evaluations reveal that these biases persist and may be amplified in more diverse populations.
- The study highlights the importance of balanced datasets, diverse annotators, and fairness-aware model design to mitigate harmful algorithmic biases.

For detailed results, see the attached paper: `BeautyBias_Hamel.pdf` and the analysis notebooks.

---

## Ethical Considerations

- The project discusses the risks of deploying biased beauty prediction models and the potential for reinforcing exclusionary cultural standards.
- Recommendations include algorithmic fairness interventions, improved data curation, and transparent validation protocols.

---

## Citation

If you use this code or analysis, please cite:


```bibtex
@misc{hamel2025beautybias,
  title     = {Analysis of Bias in AI Facial Beauty Regressors},
  author    = {Chandon Hamel},
  school    = {Regis University},
  year      = {2025},
  note      = {MSDS 640},
  address   = {Denver, CO, USA}
}
```

---

## Contact

For questions or collaboration, please contact:  
Chandon Hamel  
chamel@regis.edu

---

## License

This repository is for academic and research purposes only. Please refer to each dataset’s original license for usage terms.

---

*For more details, see the full paper and code in this repository.*
