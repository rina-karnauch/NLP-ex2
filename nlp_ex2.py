import re
import nltk
import numpy as np
from nltk.corpus import brown


def get_dictionaries(train_set, test_set):
    train_words_counter_dict, train_tags_counter_dict, train_word_tag_dict = {}, {}, {}
    word_counter_dict = {}

    for s in train_set:
        for w_t in s:
            w, t = w_t
            if w not in train_words_counter_dict.keys():
                train_words_counter_dict[w] = 0
            train_words_counter_dict[w] += 1
            if t not in train_tags_counter_dict.keys():
                train_tags_counter_dict[t] = 0
            train_tags_counter_dict[t] += 1
            if w_t not in train_word_tag_dict.keys():
                train_word_tag_dict[w_t] = 0
            train_word_tag_dict[w_t] += 1
            if w not in word_counter_dict.keys():
                word_counter_dict[w] = 0
            word_counter_dict[w] += 1

    for s in test_set:
        for w, t in s:
            if w not in word_counter_dict.keys():
                word_counter_dict[w] = 0
            word_counter_dict[w] += 1

    return train_words_counter_dict, train_tags_counter_dict, \
           train_word_tag_dict, word_counter_dict


def add_padding_to_set(train_set, test_set):
    STOP_tuple, START_tuple = ('STOP', 'STOP'), ('START', 'START')
    for s in train_set:
        s.insert(0, START_tuple)
        s.append(STOP_tuple)
    for s in test_set:
        s.insert(0, START_tuple)
        s.append(STOP_tuple)


def solution_a():
    all_sentences = [s for s in brown.tagged_sents(categories='news')]

    for s_index, s in enumerate(all_sentences):
        for w_index, w_t in enumerate(s):
            w, t = w_t
            t = re.split('[+ - *]', t)[0]
            all_sentences[s_index][w_index] = (w, t)

    train_percent_index = int(0.9 * len(all_sentences))
    train_set, test_set = all_sentences[:train_percent_index], all_sentences[train_percent_index:]
    return train_set, test_set


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


def solution_b(train_set, test_set):
    train_words_counter_dict, train_tags_counter_dict, \
    train_word_tag_counter_dict, word_counter_dict = get_dictionaries(
        train_set, test_set)

    MLE_dict = get_MLE_dict(train_words_counter_dict, train_word_tag_counter_dict)

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

    print("------------ QUESTION B ------------")
    print("error for known words:", error_known)
    print("------------------------------------")
    print("error for unknown words:", error_unknown)
    print("------------------------------------")
    print("total error:", error_total)
    print("------------------------------------")


def get_emission(word, tag, word_tag_counter_dict, tag_counter_dict, V, delta):
    w_t_counter = word_tag_counter_dict.get((word, tag), 0) + delta
    t_counter = tag_counter_dict.get(tag, 0) + (len(V) * delta)
    if w_t_counter == 0 or t_counter == 0:
        return 0
    else:
        return w_t_counter / t_counter


def get_transition_dict(train_set):
    tag_tuples_counter_dict = {}
    for s in train_set:
        for i in range(len(s) - 1):
            prev_tag, current_tag = s[i][1], s[i + 1][1]
            if prev_tag not in tag_tuples_counter_dict.keys():
                tag_tuples_counter_dict[prev_tag] = {}
            if current_tag not in tag_tuples_counter_dict[prev_tag].keys():
                tag_tuples_counter_dict[prev_tag][current_tag] = 0
            tag_tuples_counter_dict[prev_tag][current_tag] += 1

    transition_dict = {}
    for prev_tag in tag_tuples_counter_dict.keys():
        amount_prev = sum(tag_tuples_counter_dict[prev_tag].values())
        for current_tag, tuple_amount in tag_tuples_counter_dict[prev_tag].items():
            transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev

    return transition_dict


def get_transition(prev, curr, transition_dict):
    if (prev, curr) not in transition_dict.keys():
        return 0
    else:
        return transition_dict[(prev, curr)]


def viterbi(sentence, transition_dict, train_tags, train_words, word_tag_counter_dict, tag_counter_dict, delta):
    m = len(sentence)  # x0,x1,.....,xn,x_(n+1)
    t = len(train_tags)
    pi, bp = np.zeros((m - 1, t)), np.zeros((m - 1, t))

    # initial state
    for i_t, tag in enumerate(train_tags):
        pi[0, i_t] = get_transition('START', tag, transition_dict)
        bp[0, i_t] = -1  # maybe change, maybe bug

    # fill the table, without STOP
    for w_i, word in enumerate(sentence[1:-1]):
        k = w_i + 1
        for i_u, u_tag in enumerate(train_tags):
            emission = get_emission(word, u_tag, word_tag_counter_dict, tag_counter_dict, train_words, delta)
            values = emission * pi[k - 1] * [get_transition(w_tag, u_tag, transition_dict) for w_tag in train_tags]
            max_index = np.argmax(values)
            pi[k, i_u] = values[max_index]
            bp[k, i_u] = max_index

    # get maximal value with STOP transition
    max_tag_index, max_tag_value = -1, -1
    STOP_pi = [pi[-1, i_u] * get_transition(u_tag, 'STOP', transition_dict) for i_u, u_tag in enumerate(train_tags)]
    curr_max_index = np.argmax(STOP_pi)

    # trace back
    predicted_tags = np.empty(m - 2, dtype=object)
    predicted_tags[-1] = train_tags[max_tag_index]  # A[m-3]
    for k in range(m - 4, -1, -1):
        predicted_tags[k] = train_tags[int(bp[k + 1, curr_max_index])]
        curr_max_index = int(bp[k + 1, curr_max_index])
    return predicted_tags


def solution_c(train_set, test_set):
    train_words_counter_dict, train_tags_counter_dict, \
    train_word_tag_counter_dict, word_counter_dict = get_dictionaries(
        train_set, test_set)

    train_tags = list(train_tags_counter_dict.keys())
    train_words = list(train_words_counter_dict.keys())

    add_padding_to_set(train_set, test_set)
    # train set and test set were padded with START, STOP
    test_set_padded, train_set_padded = test_set, train_set
    transition_dict = get_transition_dict(train_set_padded)

    accuracy_known, known_count = 0, 0
    accuracy_unknown, unknown_count = 0, 0
    accuracy_total = 0

    for sentence in test_set_padded:
        words = [w_t[0] for w_t in sentence]
        current_tags = viterbi(words, transition_dict, train_tags, train_words,
                               train_word_tag_counter_dict, train_tags_counter_dict, delta=0)
        for w_t_i, w_t in enumerate(sentence[1:-1]):
            w, t = w_t
            if w in train_words:
                known_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_known += 1
                    accuracy_total += 1
            else:
                unknown_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_unknown += 1
                    accuracy_total += 1

    print("------------ QUESTION C ------------")
    print("error for known words:", 1 - (accuracy_known / known_count))
    print("------------------------------------")
    print("error for unknown words:", 1 - (accuracy_unknown / unknown_count))
    print("------------------------------------")
    print("total error:", 1 - (accuracy_total / (known_count + unknown_count)))
    print("------------------------------------")


def solution_d(train_set, test_set):
    train_words_counter_dict, train_tags_counter_dict, \
    train_word_tag_counter_dict, word_counter_dict = get_dictionaries(
        train_set, test_set)

    train_tags = list(train_tags_counter_dict.keys())
    train_words = list(train_words_counter_dict.keys())

    add_padding_to_set(train_set, test_set)
    # train set and test set were padded with START, STOP
    test_set_padded, train_set_padded = test_set, train_set
    transition_dict = get_transition_dict(train_set_padded)

    accuracy_known, known_count = 0, 0
    accuracy_unknown, unknown_count = 0, 0
    accuracy_total = 0

    for sentence in test_set_padded:
        words = [w_t[0] for w_t in sentence]
        current_tags = viterbi(words, transition_dict, train_tags, train_words,
                               train_word_tag_counter_dict, train_tags_counter_dict, delta=1)
        for w_t_i, w_t in enumerate(sentence[1:-1]):
            w, t = w_t
            if w in train_words:
                known_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_known += 1
                    accuracy_total += 1
            else:
                unknown_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_unknown += 1
                    accuracy_total += 1

    print("------------ QUESTION D ------------")
    print("error for known words:", 1 - (accuracy_known / known_count))
    print("------------------------------------")
    print("error for unknown words:", 1 - (accuracy_unknown / unknown_count))
    print("------------------------------------")
    print("total error:", 1 - (accuracy_total / (known_count + unknown_count)))
    print("------------------------------------")


def get_pseudo_word(word, count):
    if count > 5:
        return word
    if len(word) == 0:
        return '__empty__'
    if not word.isalpha():
        return '__sign__'
    if word.isdigit():
        return '__number__'
    if word[0].isupper():
        return '__name__'
    if len(word) > 2:
        if word[-2:] == 'ed':
            return '__past__'
    if len(word) > 3:
        if word[-3:] == 'ing':
            return '__ing__'
    if word[0] == "'" and word[-1] == "'" or word[0] == '"' and word[-1] == '"':
        return '__quote__'
    return word


def solution_e(train_set, test_set):
    train_words_counter_dict, train_tags_counter_dict, \
    train_word_tag_dict, word_counter_dict = get_dictionaries(train_set, test_set)

    new_train_set, new_test_set = [], []
    for i_s, s in enumerate(train_set):
        new_s = []
        for i_w_t, w_t in enumerate(s):
            w, t = w_t
            w_count = train_words_counter_dict.get(w, 0)
            new_s.append((get_pseudo_word(w, w_count), t))
        new_train_set.append(new_s)
    for i_s, s in enumerate(test_set):
        new_s = []
        for i_w_t, w_t in enumerate(s):
            w, t = w_t
            w_count = train_words_counter_dict.get(w, 0)
            new_s.append((get_pseudo_word(w, w_count), t))
        new_test_set.append(new_s)

    test_set, train_set = new_test_set, new_train_set

    # upon new words after pseudo conversion, calculate dictionaries
    train_words_counter_dict, train_tags_counter_dict, \
    train_word_tag_counter_dict, word_counter_dict = get_dictionaries(train_set, test_set)

    train_tags = list(train_tags_counter_dict.keys())
    train_words = list(train_words_counter_dict.keys())

    add_padding_to_set(train_set, test_set)
    # train set and test set were padded with START, STOP
    test_set_padded, train_set_padded = test_set, train_set
    transition_dict = get_transition_dict(train_set_padded)

    accuracy_known, known_count = 0, 0
    accuracy_unknown, unknown_count = 0, 0
    accuracy_total = 0

    for i, sentence in enumerate(test_set_padded):
        words = [w_t[0] for w_t in sentence]
        current_tags = viterbi(words, transition_dict, train_tags, train_words,
                               train_word_tag_counter_dict, train_tags_counter_dict, delta=0)
        for w_t_i, w_t in enumerate(sentence[1:-1]):
            w, t = w_t
            if w in train_words:
                known_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_known += 1
                    accuracy_total += 1
            else:
                unknown_count += 1
                if current_tags[w_t_i] == t:
                    accuracy_unknown += 1
                    accuracy_total += 1

    print("------------ QUESTION E ------------")
    print("error for known words:", 1 - (accuracy_known / known_count))
    print("------------------------------------")
    print("error for unknown words:", 1 - (accuracy_unknown / unknown_count))
    print("------------------------------------")
    print("total error:", 1 - (accuracy_total / (known_count + unknown_count)))
    print("------------------------------------")


if __name__ == '__main__':
    train_set, test_set = solution_a()
    # solution_b(train_set, test_set)
    # solution_c(train_set, test_set)
    # solution_d(train_set, test_set)
    solution_e(train_set, test_set)

