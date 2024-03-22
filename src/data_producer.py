import json
import random
import time

from confluent_kafka import Producer
from datasets import load_dataset

from src.configuration import data_producer_config, data_producer_topic_config


def main() -> None:
    producer1 = Producer(data_producer_config)
    producer2 = Producer(data_producer_config)

    dataset_name = 'ag_news'
    dataset = load_dataset(dataset_name)

    while True:
        sample_1_id = random.randint(0, len(dataset['test']) - 1)
        sample_2_id = random.randint(0, len(dataset['test']) - 1)

        sample_1 = dataset['test'][sample_1_id]
        sample_2 = dataset['test'][sample_2_id]

        sample_1_dict = {
            'sample': sample_1,
            'sample_id': sample_1_id,
        }

        sample_2_dict = {
            'sample': sample_2,
            'sample_id': sample_2_id,
        }

        producer1.produce(data_producer_topic_config, key='1', value=json.dumps(sample_1_dict))
        producer2.produce(data_producer_topic_config, key='1', value=json.dumps(sample_2_dict))

        producer1.flush()
        producer2.flush()

        print(f'Produced sample: {sample_1_id}')
        print(f'Produced sample: {sample_2_id}')
        time.sleep(10 + random.uniform(0, 5.0))


if __name__ == '__main__':
    main()
