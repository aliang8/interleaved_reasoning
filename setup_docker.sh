cd verl && pip install --no-deps -e .
python3 examples/data_preprocess/gsm8k.py --local_dir data/gsm8k
python3 examples/data_preprocess/math500.py --local_dir data/math500
# python3 examples/data_preprocess/gpqa.py --local_dir data/gpqa
python3 examples/data_preprocess/knights_and_knaves.py --subsets 3ppl 4ppl 5ppl --local_dir data/knights_and_knaves
python3 examples/data_preprocess/musique.py --local_dir data/musique