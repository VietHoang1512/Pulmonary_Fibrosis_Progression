export KAGGLE_USERNAME=warkingleo2000
export KAGGLE_KEY=83c04995690a76c78f7b1f94e42534c7

pip install -r requirements.txt
cd ..
mkdir logs
mkdir data
cd data

kaggle competitions download -c osic-pulmonary-fibrosis-progression
kaggle datasets download miklgr500/osic-pulmonary-fibrosis-progression-lungs-mask
# unzip osic-pulmonary-fibrosis-progression.zip
# unzip osic-pulmonary-fibrosis-progression-lungs-mask.zip