from nltk.corpus import wordnet

syns = wordnet.synsets("program")
print("Printing the Synsets: ", syns, "\n")
print("Printing the Name: ", syns[0].name())

print("-------Doing Everything for the 0th Synset Object--------")

print("Printing Lemma Objects: ", syns[0].lemmas())

print("Just the Word (of First Lemma Object):", syns[0].lemmas()[0].name())

print("Printing the definition: ", syns[0].definition())

print("Printing Examples: ", syns[0].examples())

print("Printing all lemma_names: ", wordnet.synset('plan.n.01').lemma_names())


def syn_ant(word):
    synonyms = []
    antonyms = []
    for each_synset in wordnet.synsets(word):
        for each_lemma in each_synset.lemmas():
            synonyms.append(each_lemma.name())

            if each_lemma.antonyms():
                # There is only one element in each list in the iteration
                antonyms.append(each_lemma.antonyms()[0].name())

    print("Synonyms: ", set(synonyms))
    print("Antonyms: ", set(antonyms))
    return set(synonyms), set(antonyms)


def find_similarity(word_net_object1, word_net_object2):
    similarity = word_net_object1.wup_similarity(word_net_object2)
    print(similarity)
    return similarity


if __name__ == '__main__':
    syn_ant("good")

    w1 = wordnet.synset("ship.n.01")
    w2 = wordnet.synset("boat.n.01")
    similarity = find_similarity(w1, w2)

