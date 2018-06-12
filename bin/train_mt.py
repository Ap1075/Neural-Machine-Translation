from __ import load_doc,to_pairs,norm_pairs,load_clean_dataset,save_clean_data
from __ import create_tokenizer,max_length,encode_sequences,new_gen,read_all_embeddings,define_model,mat_builder
import argparse 
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-p',meta_var="model",dest="model_path",help="Path where the model will be stored in hdf5 format post training.",type=str)
parser.add_argument("-c",meta_var="train_test_set",dest="dataset_path",help="Path to the corpus which shall be split into train and test sets.",type=str)
parser.add_argument("-w",meta_var="Word vec path", dest="word_vec",help="Path to preprocessed word embeddings",type=str)
parser.add_argument("-n",meta_var="no of epochs",dest="epochs", help='No of times whole dataset is passed through the model', type=int)
parser.add_argument("-b",metavar="batch_size",dest="batch_size",default=125, help="No of rows expected as output from generator in single call", type=int)
parser.add_argument("-s",metavar='steps_per_epoch',dest="steps_per_epoch",default=1080,help='No_of_exampleRows(train)/batch_size', type=int)
parser.add_argument("-v",metavar='validation_steps',dest="validation_steps",default=128,help='No_of_exampleRows(test)/batch_size', type=int)

args = parser.parse_args()

def trainer(model_path, dataset_path,word_vec,epochs,batch_size,steps_per_epoch,validation_steps):    
    filename = dataset_path                                      #path to corpus
    doc = load_doc(filename)
    pairs = to_pairs(doc)
    
    cleaned_pairs = norm_pairs(pairs)
    save_clean_data(cleaned_pairs,'./home/alindsharma/work/armaan/mactrans/output/english-german.pkl')
    raw_dataset = load_clean_dataset("./home/alindsharma/work/armaan/mactrans/output/english-german.pkl")
    
    np.random.shuffle(raw_dataset)
    dataset = raw_dataset[:,:]
    train, test = dataset[:135000],dataset[135000:]
    
    eng_tokenizer = create_tokenizer(dataset[:, 0])
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(dataset[:, 0])
    print('English Vocabulary Size: {}'.format(eng_vocab_size))
    print('English Max Length: %d' % (eng_length))
    # prepare german tokenizer
    ger_tokenizer = create_tokenizer(dataset[:, 1])
    ger_vocab_size = len(ger_tokenizer.word_index) + 1
    ger_length = max_length(dataset[:, 1])
    print('German Vocabulary Size: %d' % ger_vocab_size)
    print('German Max Length: %d' % (ger_length))
    
    save_clean_data(eng_tokenizer, "./home/alindsharma/work/armaan/mactrans/output/eng_tok.pkl")    
    save_clean_data(ger_tokenizer, "./home/alindsharma/work/armaan/mactrans/output/ger_tok.pkl")    
    
    dict_vars = {"ger_maxlen": ger_length, "model_path":model_path,"eng_maxlen":eng_length}
    save_clean_data(dict_vars, "./home/alindsharma/work/armaan/mactrans/output/dict_vars.pkl")
    
    # prepare training data
    trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
    trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
    #trainY = encode_output(trainY, eng_vocab_size)
    
    # prepare validation data
    testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
    testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
    #testY = encode_output(testY, eng_vocab_size)
    
    #German word embeddings
    embed = read_all_embeddings(word_vec)
    
    embedding_mat,count=mat_builder(embed,ger_tokenizer)
    
    print("*****Word vec not found for {} words*****".format(count))

    model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 512) # vocab size is total number of words, length is max sentence length
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
    filegen = model_path
    
    checkpt = ModelCheckpoint(filegen, monitor = "val_acc", verbose = 2, save_best_only = True, mode="max")
    history=model.fit_generator(generator = new_gen(trainX,trainY, batch_size, eng_vocab_size),validation_data=new_gen(testX,testY,batch_size,eng_vocab_size), epochs=epochs,steps_per_epoch=steps_per_epoch, shuffle=False, validation_steps=validation_steps , verbose=1, callbacks=[checkpt,loss_acc_hist])

trainer(args.model_path, args.dataset_path,args.word_vec,args.epochs,args.batch_size,args.steps_per_epoch,args.validation_steps)



  


	
