import numpy as np


def data_generator(annotation_lines, batch_size, char_classes, num_char_classes, max_encoder_seq_length, max_decoder_seq_length):
    n = len(annotation_lines)
    i = 0
    while True:
        encoder_input_data = []
        decoder_input_data = []
        decoder_target_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            line = annotation_lines[i]
            question = line['question']
            answer = line['answer']

            encoder_input_seq = np.zeros((max_encoder_seq_length, num_char_classes))
            decoder_input_seq = np.zeros((max_decoder_seq_length, num_char_classes))
            decoder_target_seq = np.zeros((max_decoder_seq_length, num_char_classes))

            for key, value in enumerate(question):
                encoder_input_seq[key, char_classes.index(value)] = 1
            for key, value in enumerate(answer):
                decoder_input_seq[key, char_classes.index(value)] = 1
                if key > 0:
                    decoder_target_seq[key-1, char_classes.index(value)] = 1

            encoder_input_data.append(encoder_input_seq)
            decoder_input_data.append(decoder_input_seq)
            decoder_target_data.append(decoder_target_seq)
            i = (i + 1) % n
        yield np.array(encoder_input_data), np.array(decoder_target_data)
