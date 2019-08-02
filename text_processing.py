from nltk.tokenize import sent_tokenize, word_tokenize, PunktSentenceTokenizer
from nltk.corpus import stopwords, state_union
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, RegexpParser, ne_chunk, corpus
import numpy as np


def tokenize_sentence(text):
    tokenized_sent = sent_tokenize(text)
    print(tokenized_sent, "\n")
    return tokenized_sent


def tokenize_word(sample_text_f, custom_tokenizer=False, train_text_f=None):
    if custom_tokenizer:
        custom_sent_tokenizer = PunktSentenceTokenizer(train_text_f)
        tokenized = custom_sent_tokenizer.tokenize(sample_text_f)
        print(tokenized[:10])
        return tokenized
    else:
        tokenized_word = word_tokenize(sample_text_f)
        print(tokenized_word, "\n")
        return tokenized_word


def filter_stop_words(tokenized_word, language):
    stop_words = set(stopwords.words(language))
    filtered_words = [each_word for each_word in tokenized_word if each_word not in stop_words]
    print(filtered_words, "\n")
    return filtered_words


def stem_tokens(word_tokens):
    ps = PorterStemmer()
    stemmed_words = [ps.stem(each_word) for each_word in word_tokens]
    print(stemmed_words, "\n")
    return stemmed_words


def lemmatize_tokens(word_tokens, print_lemma=False):
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(each_word) for each_word in word_tokens]
    if print_lemma:
        print(lemmatized_words, "\n")
    return lemmatized_words


def pos_tagger(tokenized_sent):
    try:
        tagged = []
        for each_sent in tokenized_sent:
            words = word_tokenize(each_sent)
            tagged.append(pos_tag(words))
        print(tagged, "\n")
        print("Length of tagged array: ", len(tagged))
        return tagged

    except Exception as e:
        print(str(e))


def chunk(pos_tagged, draw=False, print_chunked=False):
    chunk_gram_custom = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""
    chunk_gram_np = r"""NP: {<DT>?<JJ>*<NN>}"""
    chunk_parser_custom = RegexpParser(chunk_gram_custom)
    chunk_parser_np = RegexpParser(chunk_gram_np)
    # print(np.array(pos_tagged).ndim)
    if np.array(pos_tagged).ndim == 1:
        list_of_tuples_custom = []
        list_of_tuples_np = []
        for each_pos_list in pos_tagged:
            chunked_custom = chunk_parser_custom.parse(each_pos_list)
            list_of_tuples_custom.append(chunked_custom)

            chunked_np = chunk_parser_np.parse(each_pos_list)
            list_of_tuples_np.append(chunked_np)
            if print_chunked:
                print("Using Chunked Custom: ", chunked_custom)
                print("Using Chunked NP: ", chunked_np)
            if draw:
                chunked_custom.draw()
                chunked_np.draw()
        return list_of_tuples_custom, list_of_tuples_np
    else:
        chunked_custom = chunk_parser_custom.parse(pos_tagged)
        chunked_np = chunk_parser_np.parse(pos_tagged)
        if print_chunked:
            print("Using Chunked Custom: ", chunked_custom)
            print("Using Chunked NP: ", chunked_np)
        if draw:
            chunked_custom.draw()
            chunked_np.draw()
        return chunked_custom, chunked_np
    # temp = (chunked_custom, chunked_np)
    # return temp


def find_named_entity(tagged_sent, print_ne=False):
    named_entity = ne_chunk(tagged_sent)
    if print_ne:
        print(named_entity)
    return named_entity


if __name__ == '__main__':
    my_text = "Hello! This is Mr. Dipendra Karki. I am learning NLP and I am new to it. So, basically, I am" \
              " just a beginner, lol."
    my_nepali_text = "जिल्लाको एक मात्र सुकेटार विमानस्थलमा असार पहिलो सातादेखि हवाई सेवा ठप्प छ । असार " \
                     "४ गते काठमाडौंबाट " \
                     "आएको जहाजमा प्राविधिक समस्या देखिएपछि त्यो यात्रु नलिई फर्किएको थियो । " \
                     "त्यसपछि उडान नभएको नेपाल वायुसेवा " \
                     "निगमका स्टेसन इन्चार्ज राजु कार्कीले जानकारी दिए । "
    english_tokenized_sent = tokenize_sentence(my_text)
    english_tokenized_words = tokenize_word(my_text)

    nepali_tokenized_words = tokenize_word(my_nepali_text)
    nepali_filtered_words = filter_stop_words(nepali_tokenized_words, "nepali")

    # Stemming Task ----------------------------
    # nepali_stemmed_words = stem_tokens(nepali_filtered_words) # Stemming does not work for nepali word tokens
    english_stemmed_words = stem_tokens(english_tokenized_words)

    # Lemmatization Task ----------------------------
    print("Lemmatized Words:::::")
    english_lemmatized_words = lemmatize_tokens(english_tokenized_words, print_lemma=True)

    # For custom tokenization and POS Tagging
    train_text = state_union.raw("2005-GWBush.txt")
    sample_text = state_union.raw("2006-GWBush.txt")
    custom_tokenized_word = tokenize_word(sample_text, custom_tokenizer=True, train_text_f=train_text)
    custom_pos_tagged = pos_tagger(custom_tokenized_word)

    # Now we do the chunking of the custom_pos_tagged
    sample_sentence = [("the", "DT"), ("little", "JJ"), ("yellow", "JJ"),
                       ("dog", "NN"), ("barked", "VBD"), ("at", "IN"), ("the", "DT"), ("cat", "NN")]
    pos_custom, pos_np = chunk(custom_pos_tagged, draw=False, print_chunked=False)
    sample_custom, sample_np = chunk(sample_sentence, draw=False, print_chunked=False)

    # Named Entity Recognition
    sample_sentence = corpus.treebank.tagged_sents()[22]
    named_entity_tagged = find_named_entity(sample_sentence)