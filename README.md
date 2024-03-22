## Prepare model

To fine-tune target model `distilbert/distilbert-base-uncased`
```bash
python model_preparation/main.py
```
The best checkpoint will be in `data/checkpoints/best` folder.


## To run

Run the producer
```bash
source ./venv/bin/activate
export PATH=$PATH:~/.docker/bin
export PYTHONPATH=${PYTHONPATH}:.

docker compose up -d
python src/data_producer.py
```

Run the inference
```bash
source ./venv/bin/activate
export PATH=$PATH:~/.docker/bin
export PYTHONPATH=${PYTHONPATH}:.

docker compose up -d
python src/data_inference.py
```


Run the visualizer
```bash
source ./venv/bin/activate
export PATH=$PATH:~/.docker/bin
export PYTHONPATH=${PYTHONPATH}:.

docker compose up -d
streamlit run src/visualize.py
```
