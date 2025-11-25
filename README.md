# Data Science Salary Analysis with KNN and K-means Clustering

![Project Cover](https://github.com/user-attachments/assets/3fc86cc8-38d2-4655-9fdc-982554e8f91d)

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.37934%2Farca.35.1.1020-blue)](https://doi.org/10.37934/arca.35.1.1020)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project provides a comprehensive analysis of data science job salaries using two powerful machine learning techniques: **K-Nearest Neighbors (KNN)** for salary prediction and **K-Means Clustering** for identifying salary patterns. The analysis aims to uncover insights into the factors influencing salary levels and discover distinct salary segments within the data science job market.

**Key Objectives:**
- Predict salary levels based on job-related features using supervised learning
- Identify natural salary clusters and patterns using unsupervised learning
- Understand the relationship between experience level, job title, location, and compensation
- Provide data-driven insights for job seekers and employers in the data science field

**Published Research:** This work has been published in the Journal of Advanced Research in Computing and Applications. [Read the full paper here](https://doi.org/10.37934/arca.35.1.1020).

## 📊 Dataset

We use a publicly available dataset from Kaggle containing comprehensive data on data science job salaries:

- **Source**: [Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries/data)
- **Size**: 607 records
- **Features**: 12 columns
- **Time Period**: 2020-2022
- **Unique Job Titles**: 50 different roles
- **Salary Range**: $4,000 - $600,000 USD

### Dataset Features

| Column | Description | Values |
|--------|-------------|--------|
| `work_year` | Year the salary was recorded | 2020, 2021, 2022 |
| `experience_level` | Experience level of the employee | EN (Entry), MI (Mid), SE (Senior), EX (Executive) |
| `employment_type` | Type of employment | FT (Full-time), PT (Part-time), CT (Contract), FL (Freelance) |
| `job_title` | Specific job role | Data Scientist, ML Engineer, Data Analyst, etc. (50 unique titles) |
| `salary` | Salary in original currency | Varies |
| `salary_currency` | Currency of the salary | USD, EUR, GBP, INR, etc. |
| `salary_in_usd` | Standardized salary in USD | $4,000 - $600,000 |
| `employee_residence` | Country where employee resides | 2-letter country code |
| `remote_ratio` | Percentage of remote work | 0 (On-site), 50 (Hybrid), 100 (Remote) |
| `company_location` | Country where company is located | 2-letter country code |
| `company_size` | Size of the company | S (Small), M (Medium), L (Large) |

## 🔬 Methodology

The analysis employs two complementary machine learning approaches:

### 1. K-Nearest Neighbors (KNN) for Salary Prediction

**Objective**: Predict salary levels based on job-related features using supervised learning.

**Approach**:
- **Algorithm**: K-Nearest Neighbors (KNN) classification
- **Features Used**: Job title, experience level, employment type, company location, company size, remote ratio
- **Preprocessing**: 
  - Label encoding for categorical variables
  - Feature scaling/normalization
  - Train-test split for model validation
- **How it Works**: KNN identifies the K most similar data points (neighbors) to a given observation based on feature similarity. The salary level is predicted based on the majority class of these neighbors.
- **Application**: Helps estimate salary ranges for specific job profiles and understand which factors most significantly influence compensation.

### 2. K-Means Clustering for Salary Segmentation

**Objective**: Identify distinct salary clusters and patterns within the dataset using unsupervised learning.

**Approach**:
- **Algorithm**: K-Means clustering
- **Features Used**: Salary in USD, experience level, job title, company size, and other relevant features
- **Preprocessing**:
  - Feature scaling for distance-based clustering
  - Optimal K selection using elbow method or silhouette analysis
- **How it Works**: K-Means groups data points with similar characteristics into clusters, revealing natural salary segments in the data. Each cluster represents a distinct salary pattern based on job profiles.
- **Application**: Uncovers hidden salary segments, identifies typical salary ranges for different job categories, and provides insights into salary distribution across various dimensions.

## 🔍 Key Findings

### Salary Prediction with KNN

The KNN model successfully predicts salary levels by analyzing patterns in job-related features:

- **Feature Importance**: Experience level and job title are the strongest predictors of salary
- **Location Impact**: Company location significantly affects compensation, with certain regions offering premium salaries
- **Remote Work**: Remote ratio shows interesting correlations with salary levels
- **Company Size**: Larger companies tend to offer more competitive salaries for equivalent roles

### Salary Clusters with K-Means

K-Means clustering reveals distinct salary groups within the data science job market:

- **Cluster Identification**: Multiple distinct salary segments identified, each representing different job market niches
- **Salary Patterns**: Clear separation between entry-level, mid-level, senior, and executive compensation ranges
- **Geographic Variations**: Salary clusters show strong geographic patterns
- **Role Specialization**: Specialized roles (ML Engineer, Data Architect) often fall into higher salary clusters

## 📁 Project Structure

```
ds-salaries-knn-kmeans/
│
├── README.md                           # Project documentation (this file)
├── ds_salaries.csv                     # Dataset file (607 records)
├── ds_salaries_Final_Project.ipynb     # Main Jupyter notebook with analysis
│
└── .git/                               # Git version control
```

### File Descriptions

- **README.md**: Comprehensive project documentation
- **ds_salaries.csv**: Raw dataset containing 607 salary records from 2020-2022
- **ds_salaries_Final_Project.ipynb**: Complete analysis notebook including:
  - Data loading and exploration
  - Exploratory Data Analysis (EDA)
  - Data preprocessing and feature engineering
  - KNN classification implementation
  - K-Means clustering implementation
  - Visualizations and insights

## 🛠️ Technologies Used

- **Python 3.7+**: Core programming language
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms (KNN, K-Means, preprocessing)
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **Jupyter Notebook**: Interactive development environment

## 💻 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Step-by-Step Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Chanzwastaken/DataScienceSalariesWithKNNClassificationAndKMeansClustering.git
   cd DataScienceSalariesWithKNNClassificationAndKMeansClustering
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn jupyter
   ```

   Or create a `requirements.txt` file with:
   ```
   numpy>=1.19.0
   pandas>=1.1.0
   matplotlib>=3.3.0
   seaborn>=0.11.0
   scikit-learn>=0.24.0
   jupyter>=1.0.0
   ```
   
   Then install:
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

### Running the Analysis

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**
   - Navigate to `ds_salaries_Final_Project.ipynb` in the Jupyter interface
   - Click to open the notebook

3. **Run the analysis**
   - Execute cells sequentially from top to bottom
   - Use `Shift + Enter` to run each cell
   - Or use `Cell > Run All` to execute the entire notebook

### Expected Outputs

The notebook will generate:
- **Data exploration results**: Summary statistics, data types, missing values analysis
- **Visualizations**: 
  - Correlation heatmaps showing relationships between features and salaries
  - Salary distribution plots by experience level, job title, and location
  - Cluster visualizations highlighting distinct salary groups
  - KNN decision boundaries (if applicable)
- **Model performance metrics**: Accuracy, precision, recall, F1-score for KNN
- **Cluster analysis**: Cluster characteristics, centroids, and member profiles

### Troubleshooting

- **Import errors**: Ensure all dependencies are installed correctly
- **File not found**: Verify `ds_salaries.csv` is in the same directory as the notebook
- **Memory issues**: The dataset is small (607 rows), but if issues occur, try restarting the kernel
- **Visualization issues**: Ensure matplotlib backend is properly configured for your environment

## 📈 Results

### Visualization Examples

The project includes several key visualizations:

1. **Correlation Heatmaps**: Show relationships between job features and salaries
2. **Salary Distribution Plots**: Display salary ranges across different dimensions
3. **Cluster Visualizations**: Highlight distinct salary groups identified by K-Means
4. **Feature Importance**: Illustrate which factors most influence salary predictions

### Model Performance

- **KNN Classification**: Achieves competitive accuracy in predicting salary levels
- **K-Means Clustering**: Successfully identifies meaningful salary segments
- **Insights**: Provides actionable insights for both job seekers and employers

## 🔮 Future Work

### Model Improvements
- **Hyperparameter Tuning**: Optimize K value for KNN and number of clusters for K-Means using grid search
- **Feature Engineering**: Create additional features such as years of experience, industry sector, or education level
- **Feature Selection**: Apply techniques like Recursive Feature Elimination (RFE) to identify most impactful features
- **Cross-Validation**: Implement k-fold cross-validation for more robust model evaluation

### Additional Analysis
- **Alternative Algorithms**: 
  - Decision Trees and Random Forests for interpretable salary prediction
  - Gradient Boosting (XGBoost, LightGBM) for improved accuracy
  - Hierarchical clustering for alternative segmentation approaches
- **Time Series Analysis**: Analyze salary trends over the 2020-2022 period
- **Geographic Analysis**: Deep dive into regional salary variations and cost-of-living adjustments
- **Role-Specific Models**: Build specialized models for different job categories

### Data Expansion
- **Larger Dataset**: Incorporate more recent data (2023-2024) for current market insights
- **Additional Features**: Include education level, years of experience, specific skills, company industry
- **Real-time Updates**: Implement pipeline for continuous data collection and model updates

### Deployment
- **Web Application**: Create interactive dashboard for salary exploration and prediction
- **API Development**: Build REST API for programmatic access to predictions
- **Mobile App**: Develop mobile application for on-the-go salary insights

## 📚 Citation

If you use this project in your research or work, please cite the following publication:

### APA Format
```
Leveraging Correlation and Clustering: An Exploration of Data Scientist Salaries. (2024). 
Journal of Advanced Research in Computing and Applications, 35(1), 10-20. 
https://doi.org/10.37934/arca.35.1.1020
```

### BibTeX Format
```bibtex
@article{datasalaries2024,
  title={Leveraging Correlation and Clustering: An Exploration of Data Scientist Salaries},
  journal={Journal of Advanced Research in Computing and Applications},
  volume={35},
  number={1},
  pages={10--20},
  year={2024},
  doi={10.37934/arca.35.1.1020},
  url={https://doi.org/10.37934/arca.35.1.1020}
}
```

### Publication Link
**Full Paper**: [https://doi.org/10.37934/arca.35.1.1020](https://doi.org/10.37934/arca.35.1.1020)

## 📄 License

This project is available under the MIT License. See the LICENSE file for more details.

## 👤 Author

**Chanzwastaken**
- GitHub: [@Chanzwastaken](https://github.com/Chanzwastaken)
- Repository: [DataScienceSalariesWithKNNClassificationAndKMeansClustering](https://github.com/Chanzwastaken/DataScienceSalariesWithKNNClassificationAndKMeansClustering)

## 🙏 Acknowledgments

- Dataset provided by [Kaggle - Data Science Job Salaries](https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries/data)
- Published in the Journal of Advanced Research in Computing and Applications
- Built with open-source tools and libraries from the Python data science ecosystem

---

**⭐ If you find this project useful, please consider giving it a star on GitHub!**
