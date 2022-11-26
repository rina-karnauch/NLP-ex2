import re
import nltk
import numpy as np
from nltk.corpus import brown



def solution_a():
    tagged_sents = [s for s in brown.tagged_sents(categories='news')]

    for s_index, s in enumerate(tagged_sents):
        for w_index, w_t in enumerate(s):
            w, t = w_t
            t = re.split('[+ - *]', t)[0]
            tagged_sents[s_index][w_index] = (w, t)

    ninenty_percent_index = int(0.9 * len(tagged_sents))
    train_set, test_set = tagged_sents[:ninenty_percent_index], tagged_sents[ninenty_percent_index:]
    return train_set, test_set


def solution_b(train_set, test_set):
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

    train_word_counter_dict, train_word_tag_counter_dict = get_dictionaries(train_set)
    unknown_word_tag_dict = get_all_unknown_word_tag_dict(train_word_counter_dict, test_set)
    MLE_dict = get_MLE_dict(train_word_counter_dict, train_word_tag_counter_dict)
    get_error_rate(test_set, MLE_dict, unknown_word_tag_dict)
    return train_word_counter_dict, train_word_tag_counter_dict, unknown_word_tag_dict


def get_emission(word, tag, train_words, emission_dict):
    if (word, tag) not in emission_dict.keys():
        return 1 if (word not in train_words and tag == 'NN') else 0
    else:
        return emission_dict[(word, tag)]


def get_transition(prev, curr, transition_dict):
    return transition_dict.get((prev, curr), 0)


# def viterbi(s, S, train_words, transition_dict, emission_dict):
#     n, t = len(s), len(S)
#     S_0 = ['START']
#     pi, bp = np.zeros(shape=(n - 1, t)), np.empty(shape=(n - 1, t), dtype=object)  # probabilities
#
#     # init arrays
#     for i in range(t):
#         pi[0, i] = 1
#         bp[0, i] = 'START'
#
#     # calculate matrices, n is calculated later
#     for k in range(1, n - 1):
#         x_k = s[k][0]  # x1->y1 ... xn->yn
#         # now we fill bp[k-1]
#         for i_v, v in enumerate(S):
#             S_k_1 = S if k > 1 else S_0
#             arg_max, max = 'NN', 0
#             for i_u, u in enumerate(S_k_1):
#                 pi_k_1_u = pi[k - 1, i_u]
#                 transition_v_u = get_transition(u, v, transition_dict)
#                 emission_x_k_v = get_emission(x_k, v, train_words, emission_dict)
#                 pi_k_v = pi_k_1_u * transition_v_u * emission_x_k_v
#                 if pi_k_v > max:
#                     max = pi_k_v
#                     arg_max = u
#             pi[k, i_v] = max
#             bp[k, i_v] = arg_max
#
#     # backtrack
#     predicted_tags = np.empty(shape=(n - 2), dtype=object)
#     arg_max, tag_max = get_max_pi_q(pi, n - 2, S, transition_dict)
#     predicted_tags[-1] = (tag_max, arg_max)  # y_n....y_1 # x_n-1
#     # START, y1,y2,...,y(n-2),y(n-1)=STOP
#     # predicted_tags = 0 ,..., n-3=already have
#     for i in range(n - 4, -1, -1):
#         predicted_tags[i] = (bp[i + 1, predicted_tags[i + 1][1]], predicted_tags[i + 1][1])
#     word_tags = [tag[0] for tag in predicted_tags]
#     return word_tags
#
#
# def get_max_pi_q(pi, n, S, transition_dict):
#     # final row for STOP
#     arg_max, tag_max, max = 0, 'NN', 0
#     k, stop = n, 'STOP'
#     for i_u, u in enumerate(S):
#         pi_c = pi[n, i_u]
#         transition_v_u = get_transition(u, stop, transition_dict)
#         current_max = pi_c * transition_v_u
#         if current_max > max:
#             max = current_max
#             arg_max = i_u
#             tag_max = u
#     return arg_max, tag_max


# def solution_c(train_set, test_set):
#     def get_dictionaries(train_padded):
#         word_counter, tag_counter, tag_tuples_counter, word_tag_counter = dict(), dict(), dict(), dict()
#         transition_dict, emission_dict = dict(), dict()
#
#         for s in train_padded:
#             window_s = s[1:]
#             for i, w_t in enumerate(window_s):
#                 index = i + 1
#                 w, t = w_t
#                 if w not in word_counter.keys():
#                     word_counter[w] = 0
#                 word_counter[w] += 1
#                 if t not in tag_counter.keys():
#                     tag_counter[t] = 0
#                 tag_counter[t] += 1
#                 if w_t not in word_tag_counter.keys():
#                     word_tag_counter[w_t] = 0
#                 word_tag_counter[w_t] += 1
#                 previous_tag = s[index - 1][1]
#                 if previous_tag not in tag_tuples_counter.keys():
#                     tag_tuples_counter[previous_tag] = {t: 0}
#                 elif t not in tag_tuples_counter[previous_tag].keys():
#                     tag_tuples_counter[previous_tag] = {t: 0}
#                 tag_tuples_counter[previous_tag][t] += 1
#                 # y2|y1 = #(y1,y2) / #(y1)
#                 # x1|y1 = #(x1,y1) / #(y1)
#         tag_counter['START'] = len(train_padded)
#
#         for c_w_t in word_tag_counter.keys():
#             c_w, c_t = c_w_t
#             emission_dict[c_w_t] = word_tag_counter[c_w_t] / tag_counter[c_t]
#         for prev_tag in tag_tuples_counter.keys():
#             # amount_prev = tag_counter[prev_tag]
#             amount_prev = sum(tag_tuples_counter[prev_tag].values())
#             for current_tag, tuple_amount in tag_tuples_counter[prev_tag].items():
#                 transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev
#         return transition_dict, emission_dict, tag_counter.keys(), word_counter.keys()
#
#     train_padded, test_padded = add_padding_to_sets(train_set, test_set)
#     transition_dict, emission_dict, train_tags, train_words = get_dictionaries(train_padded)
#
#     accuracy_known, known_count = 0, 0
#     accuracy_unknown, unknown_count = 0, 0
#     accuracy_total = 0
#     for i, sentence in enumerate(test_set):
#         words = [w_t[0] for w_t in sentence]
#         current_tags = viterbi(words, train_tags, train_words, transition_dict, emission_dict)
#         for j, w_t in enumerate(sentence[1:-1]):
#             w, t = w_t
#             if w in train_words:
#                 known_count += 1
#                 if current_tags[j] == t:
#                     accuracy_known += 1
#                     accuracy_total += 1
#             else:
#                 unknown_count += 1
#                 if current_tags[j] == t:
#                     accuracy_unknown += 1
#                     accuracy_total += 1
#     print("------------ QUESTION C ------------")
#     print("error for known words:", 1 - (accuracy_known / known_count))
#     print("------------------------------------")
#     print("error for unknown words:", 1 - (accuracy_unknown / unknown_count))
#     print("------------------------------------")
#     print("total error:", 1 - (accuracy_total / (known_count + unknown_count)))
#     print("------------------------------------")
#
#
# def add_padding_to_sets(train_set, test_set):
#     STOP_tuple, START_tuple = ('STOP', 'STOP'), ('START', 'START')
#     for s in train_set:
#         s.insert(0, START_tuple)
#         s.append(STOP_tuple)
#     for s in test_set:
#         s.insert(0, START_tuple)
#         s.append(STOP_tuple)
#     return train_set, test_set
#
#

def add_padding_to_set(train_set, test_set):
    STOP_tuple, START_tuple = ('STOP', 'STOP'), ('START', 'START')
    for s in train_set:
        s.insert(0, START_tuple)
        s.append(STOP_tuple)
    for s in test_set:
        s.insert(0, START_tuple)
        s.append(STOP_tuple)
    return train_set, test_set

def solution_c(train_set, test_set):
    def get_dictionaries(train_padded):
        word_counter, tag_counter, tag_tuples_counter, word_tag_counter = dict(), dict(), dict(), dict()
        transition_dict, emission_dict = dict(), dict()

        for s in train_padded:
            window_s = s[:-1]
            for i, prev_tuple in enumerate(window_s):
                prev_word, prev_tag = window_s[i]
                current_word, current_tag = window_s[i + 1]

                if prev_word not in word_counter.keys():
                    word_counter[prev_word] = 0
                word_counter[prev_word] += 1

                if prev_tag not in tag_counter.keys():
                    tag_counter[prev_tag] = 0
                tag_counter[prev_tag] += 1

                if prev_tuple not in word_tag_counter.keys():
                    word_tag_counter[prev_tuple] = 0
                word_tag_counter[prev_tuple] += 0

                if (prev_tag, current_tag) not in tag_tuples_counter.keys():
                    if prev_tag not in tag_tuples_counter.keys():
                        tag_tuples_counter[prev_tag] = {current_tag: 0}
                    elif current_tag not in tag_tuples_counter[prev_tag].keys():
                        tag_tuples_counter[prev_tag] = {current_tag: 0}
                tag_tuples_counter[prev_tag][current_tag] += 1

        for c_w_t in word_tag_counter.keys():
            c_w, c_t = c_w_t
            emission_dict[c_w_t] = word_tag_counter[c_w_t] / tag_counter[c_t]
        for prev_tag in tag_tuples_counter.keys():
            amount_prev = sum(tag_tuples_counter[prev_tag].values())
            for current_tag, tuple_amount in tag_tuples_counter[prev_tag].items():
                transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev
        return transition_dict, emission_dict, tag_counter.keys(), word_counter.keys()

    train_padded, test_padded = add_padding_to_set(train_set, test_set)
    transition_dict, emission_dict, train_tags, train_words = get_dictionaries(train_padded)


def viterbi(sentence, transition_dict, emission_dict, train_words, train_tags):
    word_number = len(sentence)




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
    print("------------ QUESTION B ------------")
    print("error for known words:", error_known)
    print("------------------------------------")
    print("error for unknown words:", error_unknown)
    print("------------------------------------")
    print("total error:", error_total)
    print("------------------------------------")


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
    # solution_b(train_set, test_set)
    solution_c(train_set, test_set)
