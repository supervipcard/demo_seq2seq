import json
import numpy as np
from keras.optimizers import Adam, SGD
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from keras.layers import K
import seq2seq
from seq2seq.models import AttentionSeq2Seq, Seq2Seq

from network import net
from utils import data_generator


def train():
    annotation_path = 'train_data.json'
    classes_path = 'char_data.json'
    weights_path = 'model_data/weights.h5'
    log_dir = 'logs/'
    batch_size = 4

    char_classes = json.load(open(classes_path, 'r', encoding='utf8'))
    num_char_classes = len(char_classes)

    lines = json.load(open(annotation_path, 'r', encoding='utf8'))
    max_encoder_seq_length = max(len(i['question']) for i in lines)
    max_decoder_seq_length = max(len(i['answer']) for i in lines)

    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss', save_weights_only=True, save_best_only=False, period=3)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    model = AttentionSeq2Seq(output_dim=num_char_classes, output_length=max_decoder_seq_length, input_dim=num_char_classes, input_length=max_encoder_seq_length, hidden_dim=32, depth=(1, 1))
    model.compile(loss='mse', optimizer='rmsprop')

    # model = net(num_char_classes)
    # model.compile(loss="categorical_crossentropy", optimizer=Adam())

    # model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    model.fit_generator(generator=data_generator(lines[:num_train], batch_size, char_classes, num_char_classes, max_encoder_seq_length, max_decoder_seq_length),
                        steps_per_epoch=max(1, num_train // batch_size),
                        validation_data=data_generator(lines[num_train:], batch_size, char_classes, num_char_classes, max_encoder_seq_length, max_decoder_seq_length),
                        validation_steps=max(1, num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[logging, checkpoint, reduce_lr, early_stopping])
    model.save_weights(log_dir + 'trained_weights_final.h5')


if __name__ == '__main__':
    train()
