Step-by-step dataset preprocessing:

1. Create train.json and dev.json manifests

2. Run
```
python src/utils/fastspeech2/create_txt_transcipts.py --manifest-path {dataset}/{train,dev}.json
```

3. Run
```
python src/utils/fastspeech2/extract_energy_pitch.py \
--manifest-path {train,dev}.json \
--root-path {dataset_path} \
--config-path src/utils/fastspeech2/preprocessing_config_22050.yaml
```

4. Create conda env with mfa
```
conda create -n aligner -c conda-forge openblas python=3.8 openfst pynini ngram baumwelch
conda activate aligner
pip install tgt torch
conda install -c conda-forge montreal-forced-aligner
```

5. Download CMUdict
```
wget https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict
```

6. Patch CMUdict
```
sed 's/\ \#.*//' cmudict.dict > uncommented_cmudict.dict
```

7. Download acoustic model
```
conda activate aligner
mfa model download acoustic english
```

8. Align dataset
```
mfa align --clean {dataset} uncommented_cmudict.dict english {dataset}/alignments
```

9. Create phomene mappings
```
python src/utils/fastspeech2/create_token2idx_dict.py \
--dict-path uncommented_cmudict.dict \
--mapping-path {dataset}/mappings.json
```
