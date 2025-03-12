# 🎵 **FMA ML Project: Advanced Data Exploration & Model Training**  

> **Extending the Free Music Archive (FMA) dataset** to train ML models and explore deep data insights.  

 **Project Focus:**  
🔹 In-depth dataset analysis 🔹 Advanced machine learning 🔹 High-performance genre classification  

---  

## 📂 **Dataset & Research**  

The **Free Music Archive (FMA) dataset** serves as the foundation for this project. It provides metadata, features, and MP3-encoded audio files for over 100,000 tracks.  

📁 **Access the original dataset & research:**  

- 🔗 **[FMA Dataset on GitHub][FMA_GitHub]**  
- 📄 **[ISMIR 2017 Research Paper][FMA_Paper]**  
- 🎵 **[Free Music Archive Website][FMA_Website]**  

[FMA_GitHub]: https://github.com/mdeff/fma  
[FMA_Paper]: https://arxiv.org/abs/1612.01840  
[FMA_Website]: https://freemusicarchive.org  

---  

## **Dataset Overview**  

### **Metadata & Features**  

All metadata and extracted features are stored in **[`fma_metadata.zip`]** (342 MB).  

| File | Description |
|------|------------|
| `tracks.csv` | Metadata for 106,574 tracks (ID, title, artist, genres, play counts) |
| `genres.csv` | 163 genres with hierarchical structure |
| `features.csv` | Audio features extracted with [Librosa] |
| `echonest.csv` | Features from [Echonest] (now [Spotify]) for 13,129 tracks |  

[Librosa]: https://librosa.org/  
[Echonest]: https://the.echonest.com/  

### **Audio Data (MP3 Format)**  

| Dataset | Tracks | Duration | Size |
|---------|--------|----------|------|
| [`fma_small.zip`] | 8,000 | 30s | 7.2 GB |
| [`fma_medium.zip`] | 25,000 | 30s | 22 GB |
| [`fma_large.zip`] | 106,574 | 30s | 93 GB |
| [`fma_full.zip`] | 106,574 | Full | 879 GB |  

[`fma_metadata.zip`]: https://os.unil.cloud.switch.ch/fma/fma_metadata.zip  
[`fma_small.zip`]: https://os.unil.cloud.switch.ch/fma/fma_small.zip  
[`fma_medium.zip`]: https://os.unil.cloud.switch.ch/fma/fma_medium.zip  
[`fma_large.zip`]: https://os.unil.cloud.switch.ch/fma/fma_large.zip  
[`fma_full.zip`]: https://os.unil.cloud.switch.ch/fma/fma_full.zip  

---  

##  **Project Code**  

| Notebook/Script | Description |
|----------------|------------|
| [`usage.ipynb`] | Load datasets, develop, train, and test ML models |
| [`analysis.ipynb`] | Dataset exploration & visualization |
| [`baselines.ipynb`] | Baseline models for genre recognition |
| [`features.py`] | Extract features from audio files |
| [`webapi.ipynb`] | Query the FMA API |
| [`creation.ipynb`] | Generate dataset files (`tracks.csv`, `genres.csv`) |
| [`utils.py`] | Helper functions & utilities |  

[`usage.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/usage.ipynb  
[`analysis.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/analysis.ipynb  
[`baselines.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/baselines.ipynb  
[`features.py`]: features.py  
[`webapi.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/webapi.ipynb  
[`creation.ipynb`]: https://nbviewer.jupyter.org/github/mdeff/fma/blob/outputs/creation.ipynb  
[`utils.py`]: utils.py  

---  

## ⚙️ **Environment Setup**  

To ensure **consistency** across all team members, we use a **Conda environment** (`fma_ml`).  

### **Step 1: Install Conda**  

Ensure Conda is installed. If not, install:  

- **[Miniforge (Recommended)](https://github.com/conda-forge/miniforge)**  
- **[Anaconda](https://www.anaconda.com/products/distribution)**  

Verify Conda installation:  

```sh
conda --version
```  

###  **Step 2: Clone and Set Up the Environment**  

```sh
# Clone the repository
git clone https://github.com/ahmadagah/fma-ml-project
cd ml-fma-finalProject

# Create the Conda environment (Python 3.9.21)
conda env create -f environment.yml

# Activate the environment
conda activate fma_ml
```  

###  **Step 3: Register Jupyter Kernel**  

Jupyter is already included in `environment.yml`. No need for a global install.  

```sh
python -m ipykernel install --user --name=fma_ml --display-name "Python (fma_ml)"
```  

### 📓 **Step 4: Run Jupyter Notebook**  

```sh
jupyter notebook
```
Go to **Kernel → Change Kernel → Python (fma_ml)** to ensure the correct environment is active.  

---  

##  **Downloading & Verifying Data**  

Download, verify integrity, and unzip archives:  

```sh
cd data

curl -O https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_medium.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_large.zip
curl -O https://os.unil.cloud.switch.ch/fma/fma_full.zip

echo "f0df49ffe5f2a6008d7dc83c6915b31835dfe733  fma_metadata.zip" | sha1sum -c -
echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip" | sha1sum -c -
echo "c67b69ea232021025fca9231fc1c7c1a063ab50b  fma_medium.zip" | sha1sum -c -
echo "497109f4dd721066b5ce5e5f250ec604dc78939e  fma_large.zip" | sha1sum -c -
echo "0f0ace23fbe9ba30ecb7e95f763e435ea802b8ab  fma_full.zip" | sha1sum -c -

unzip fma_metadata.zip
unzip fma_small.zip
unzip fma_medium.zip
unzip fma_large.zip
unzip fma_full.zip

cd ..

```  

🔹 If decompression errors occur, try **[7zip](https://www.7-zip.org)**.  

---  

##  **Final Setup**  

Create a `.env` file at the project root with:  

```sh

AUDIO_DIR=./data/fma_small/  # Path to decompressed fma_*.zip files
FMA_KEY=MYKEY  # (Only if querying the Free Music Archive API)
```  

Now, launch Jupyter and get started:  

```sh
jupyter notebook
make usage.ipynb
```  

 **You're all set!** Happy coding! 
