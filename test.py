from AbstractPredictor import AbstractPredictor
from PredictorRNN import PredictorRNN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

abstract_model = AbstractPredictor("C:\\Users\\ahettiarac\\csv_data\\Reviews\\Reviews.csv")
#preprocessed_text,word_dict_text = abstract_model.preprocess_text("Text")
#preprocessed_summary,word_dict_summary = abstract_model.preprocess_text("Summary")

#abstract_model.data['cleaned_text'] = preprocessed_text
#abstract_model.data['cleaned_summary'] = preprocessed_summary

#abstract_model.data.replace('', np.nan, inplace=True)
#abstract_model.data.dropna(axis=0, inplace=True)

#text_word_count = []
#summary_word_count = []

#for i in abstract_model.data['cleaned_text']:
#    text_word_count.append(len(i.split()))

#for i in abstract_model.data['cleaned_summary']:
#    summary_word_count.append(len(i.split()))

#length_df = pd.DataFrame({'text': text_word_count, 'summary': summary_word_count})
#length_df.hist(bins=30)
#plt.show()

#x_train,x_test,y_train,y_test = abstract_model.pre_arrange_text_data(abstract_model.data)
#x_train,x_test,y_train,y_test = abstract_model.arrange_model_data(x_train, x_test, y_train, y_test)
#x_train,x_test,y_train,y_test = abstract_model.remove_empty_rows(x_train,x_test, y_train, y_test)        

#print(x_train.shape)
#print("==============")
#print(y_train.shape)
#print("==============")
#print(x_test.shape)
#print("==============")
#print(y_test.shape)

model = PredictorRNN(abstract_model.max_text_length, abstract_model.max_summary_length, abstract_model.vocab_size,abstract_model.x_vocab_size)
test_seq,y_tokenizer,x_tokenizer = abstract_model.get_input_seq("salt separate area pain makes hard regulate salt putting like salt go ahead get product")
model.x_vocab_size = x_tokenizer.num_words + 1
model.vocab_size = y_tokenizer.num_words + 1
model.build_model()
encoder_model,decoder_model = model.get_inference_model(abstract_model.max_text_length)
print("Predicted summary:",model.decode_sequence(test_seq, encoder_model, decoder_model,y_tokenizer.word_index,abstract_model.max_summary_length,y_tokenizer.index_word))
#model.train_model(x_train, x_test, y_train, y_test, word_dict_summary)