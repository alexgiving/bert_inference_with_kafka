import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from confluent_kafka import Consumer

from src.configuration import (data_processor_config,
                               data_processor_topic_config)


def main():
    topic_consume = [data_processor_topic_config]
    conf_consume = {**data_processor_config, 'group.id': 'data_visualizers'}
    consumer = Consumer(conf_consume)
    consumer.subscribe(topic_consume)

    st.set_page_config(
        page_title='Real-Time Data Dashboard',
        layout='wide',
    )

    container_processed_samples = st.container(border=True)
    container_processed_samples.title('Counter')
    container_processed_samples = container_processed_samples.empty()

    container_accuracies = st.container(border=True)
    container_accuracies.title('Accuracies')
    container_accuracies = container_accuracies.empty()

    total_result = defaultdict(list)

    while True:
        msg = consumer.poll(timeout=1000)

        if msg is not None:
            container_processed_samples = container_processed_samples.empty()
            data = json.loads(msg.value().decode('utf-8'))

            predicted = data['predicted']
            reference = data['reference']

            total_result[reference].append(predicted)

            text_info = '\n'.join(
                [f'Class {class_name} : {len(results)} samples' for class_name, results in total_result.items()]
            )
            container_processed_samples.write(text_info)

            column_classes = list(total_result.keys())
            accuracies = [
                int(sum(np.array(predicted) == reference)/len(predicted)*100) for reference, predicted in total_result.items()
            ]
            print(f'Accuracies: {accuracies}', flush=True)

            fig, ax = plt.subplots()
            fig.set_size_inches(4, 4)
            bar_container = ax.bar(column_classes, accuracies)
            ax.set(ylabel='Correct answers, %', title='Per class Accuracy', ylim=(0, 100))
            ax.bar_label(bar_container, fmt='{:,.0f}')
            container_accuracies.pyplot(fig, use_container_width=False)
            plt.close()


if __name__ == '__main__':
    main()