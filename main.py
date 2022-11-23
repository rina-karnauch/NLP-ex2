import re
from nltk.corpus import brown


def solution_a():
    tagged_sents = [s for s in brown.tagged_sents(categories='news')]

    for s_index, s in enumerate(tagged_sents):
        for w_index, w_t in enumerate(s):
            w, t = w_t
            t = re.split(r'[*-+]', t)[0]
            tagged_sents[s_index][w_index] = (w, t)

    ninenty_percent_index = int(0.9 * len(tagged_sents))
    train_set, test_set = tagged_sents[:ninenty_percent_index], tagged_sents[ninenty_percent_index:]
    return train_set, test_set


def solution_b(train_set, test_set):
    train_word_counter_dict, train_word_tag_counter_dict = get_dictionaries(train_set)
    unknown_word_tag_dict = get_all_unknown_word_tag_dict(train_word_counter_dict, test_set)
    MLE_dict = get_MLE_dict(train_word_counter_dict, train_word_tag_counter_dict)
    get_error_rate(test_set, MLE_dict, unknown_word_tag_dict)


def get_error_rate(test_set, MLE_dict, unknown_word_tag_dict):
    known_accuracy_counter = 0
    unknown_accuracy_counter = 0
    test_set_known_words_counter = 0
    test_set_unknown_words_counter = 0
    for s in test_set:
        for w_t in s:
            w, t = w_t
            if w in MLE_dict.keys():
                if MLE_dict[w] == t:
                    known_accuracy_counter += 1
                test_set_known_words_counter += 1
            else:
                if t == 'NN':
                    unknown_accuracy_counter += 1
                test_set_unknown_words_counter += 1
    error_known = 1 - (known_accuracy_counter / test_set_known_words_counter)
    error_unknown = 1 - (unknown_accuracy_counter / test_set_unknown_words_counter)
    error_total = 1 - ((known_accuracy_counter + unknown_accuracy_counter) / (
            test_set_known_words_counter + test_set_unknown_words_counter))
    print("-----------------------------")
    print("error for known words:", error_known)
    print("-----------------------------")
    print("error for unknown words:", error_unknown)
    print("-----------------------------")
    print("total error:", error_total)
    print("-----------------------------")


def get_MLE_dict(train_word_counter_dict, train_word_tag_counter_dict):
    prob_dict = dict()
    for w_t, count_w_t in train_word_tag_counter_dict.items():
        w, t = w_t
        if w not in prob_dict.keys():
            prob_dict[w] = {t: count_w_t / train_word_counter_dict[w]}
        else:
            prob_dict[w][t] = count_w_t / train_word_counter_dict[w]
    MLE_dict = dict()
    for w in prob_dict.keys():
        word_tags_dict = prob_dict[w]
        MLE_dict[w] = max(word_tags_dict, key=word_tags_dict.get)
    return MLE_dict


def get_dictionaries(train_set):
    word_counter_dict = dict()
    word_tag_counter_dict = dict()
    for s in train_set:
        for w_t in s:
            w, t = w_t
            if w not in word_counter_dict.keys():
                word_counter_dict[w] = 0
            word_counter_dict[w] += 1
            if w_t not in word_tag_counter_dict.keys():
                word_tag_counter_dict[w_t] = 0
            word_tag_counter_dict[w_t] += 1
    return word_counter_dict, word_tag_counter_dict


def get_all_unknown_word_tag_dict(train_words_dict, test_set):
    unknown_words_tag = 'NN'
    unknown_word_tag_dict = dict()
    known_words = train_words_dict.keys()
    for s in test_set:
        for w_t in s:
            w, t = w_t
            if w not in known_words:
                unknown_word_tag_dict[(w, unknown_words_tag)] = 1
    return unknown_word_tag_dict


if __name__ == '__main__':
    train_set, test_set = solution_a()
    solution_b(train_set, test_set)
