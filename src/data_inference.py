import json
from pathlib import Path

import torch
from confluent_kafka import Consumer, Producer

from model_preparation.main import prepare_model_tokenizer
from src.configuration import (data_processor_config,
                               data_processor_topic_config,
                               data_producer_config,
                               data_producer_topic_config)


def main() -> None:
    topic_consume = [data_producer_topic_config]
    conf_consume = {**data_producer_config, 'group.id': 'data_processors'}
    consumer = Consumer(conf_consume)
    consumer.subscribe(topic_consume)

    producer = Producer(data_processor_config)

    checkpoint_path = Path('data/checkpoints/best')
    model_name = 'distilbert/distilbert-base-uncased'

    num_classes = 4
    model, tokenizer = prepare_model_tokenizer(model_name, num_classes, ckpt_path=checkpoint_path)
    model.eval()

    while True:
        msg = consumer.poll(timeout=1000)

        if msg is not None:
            data = json.loads(msg.value().decode('utf-8'))
            sample = data['sample']

            inputs = tokenizer(sample['text'], truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()

            result_data = {
                'predicted': predicted_class_id,
                'reference': sample['label']
            }

            producer.produce(data_processor_topic_config, key='1', value=json.dumps(result_data))
            producer.flush()

            sample_id = data['sample_id']
            print(f'Processed: {sample_id}', flush=True)


if __name__ == '__main__':
    main()
