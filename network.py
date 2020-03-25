from keras.layers import Input, Dense, LSTM, GRU
from keras.models import Model


def net(num_char_classes):
    encoder_inputs = Input(shape=(None, num_char_classes))  # 编码器输入序列（问题序列），num_char_classes是字符集总量
    encoder_outputs, state_h, state_c = LSTM(256, return_state=True)(encoder_inputs)  # 经过编码器的循环层，只需要状态，不需要输出
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=[encoder_inputs], outputs=encoder_states)  # 编码器模型

    # 训练
    decoder_inputs = Input(shape=(None, num_char_classes))  # 解码器输入序列（答案序列），要有一个起始字符和一个终止字符
    decoder_outputs, state_h, state_c = LSTM(256, return_sequences=True, return_state=True)(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = Dense(num_char_classes, activation='softmax')(decoder_outputs)  # 解码器输出序列（也是答案序列）比解码器输入序列少一个起始字符，相当于快一个时序
    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[decoder_outputs])

    # 从编码器中得到状态值（上下文），作为解码器循环层的初始状态。
    # 预测流程：输入起始字符，得到答案序列的第一个字符a，将字符a再作为解码器的输入，输出答案序列的第二个字符，依次类推，直到输出终止字符
    decoder_inputs = Input(shape=(None, num_char_classes))
    decoder_inputs_state_h = Input(shape=(256,))
    decoder_inputs_state_c = Input(shape=(256,))
    decoder_inputs_states = [decoder_inputs_state_h, decoder_inputs_state_c]
    decoder_outputs, state_h, state_c = LSTM(256, return_sequences=True, return_state=True)(decoder_inputs, initial_state=decoder_inputs_states)
    decoder_outputs_states = [state_h, state_c]
    decoder_outputs = Dense(num_char_classes, activation='softmax')(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_inputs_states, outputs=[decoder_outputs] + decoder_outputs_states)

    return model, encoder_model, decoder_model


def main():
    model, encoder_model, decoder_model = net(num_char_classes=1000)
    print(model.summary())


if __name__ == '__main__':
    main()
