# import re
# import nltk
# import numpy as np
# from nltk.corpus import brown
#
#
# def solution_a():
#     tagged_sents = [s for s in brown.tagged_sents(categories='news')]
#
#     for s_index, s in enumerate(tagged_sents):
#         for w_index, w_t in enumerate(s):
#             w, t = w_t
#             t = re.split('[+ - *]', t)[0]
#             tagged_sents[s_index][w_index] = (w, t)
#
#     ninenty_percent_index = int(0.9 * len(tagged_sents))
#     train_set, test_set = tagged_sents[:ninenty_percent_index], tagged_sents[ninenty_percent_index:]
#     return train_set, test_set
#
#
# def solution_b(train_set, test_set):
#     def get_dictionaries(train_set):
#         word_counter_dict = dict()
#         word_tag_counter_dict = dict()
#         for s in train_set:
#             for w_t in s:
#                 w, t = w_t
#                 if w not in word_counter_dict.keys():
#                     word_counter_dict[w] = 0
#                 word_counter_dict[w] += 1
#                 if w_t not in word_tag_counter_dict.keys():
#                     word_tag_counter_dict[w_t] = 0
#                 word_tag_counter_dict[w_t] += 1
#         return word_counter_dict, word_tag_counter_dict
#
#     train_word_counter_dict, train_word_tag_counter_dict = get_dictionaries(train_set)
#     MLE_dict = get_MLE_dict(train_word_counter_dict, train_word_tag_counter_dict)
#     get_error_rate(test_set, MLE_dict)
#     return train_word_counter_dict, train_word_tag_counter_dict
#
#
# def get_emission(word, tag, train_words, emission_dict):
#     if (word, tag) not in emission_dict.keys():
#         return 1 if (word not in train_words and tag == 'NN') else 0
#     else:
#         return emission_dict[(word, tag)]
#
#
# def get_transition(prev, curr, transition_dict):
#     return transition_dict.get((prev, curr), 0)
#
#
# def add_padding_to_set(train_set, test_set):
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
# def get_c_dictionaries(train_padded):
#     word_counter, tag_counter, tag_tuples_counter, word_tag_counter = dict(), dict(), dict(), dict()
#     transition_dict, emission_dict = dict(), dict()
#
#     for s in train_padded:
#         window_s = s[:-1]
#         for i, prev_tuple in enumerate(window_s):
#             prev_word, prev_tag = s[i]
#             current_word, current_tag = s[i + 1]
#
#             if prev_word not in word_counter.keys():
#                 word_counter[prev_word] = 0
#             word_counter[prev_word] += 1
#
#             if prev_tag not in tag_counter.keys():
#                 tag_counter[prev_tag] = 0
#             tag_counter[prev_tag] += 1
#
#             if prev_tuple not in word_tag_counter.keys():
#                 word_tag_counter[prev_tuple] = 0
#             word_tag_counter[prev_tuple] += 0
#
#             if (prev_tag, current_tag) not in tag_tuples_counter.keys():
#                 if prev_tag not in tag_tuples_counter.keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#                 elif current_tag not in tag_tuples_counter[prev_tag].keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#             tag_tuples_counter[prev_tag][current_tag] += 1
#
#     for c_w_t in word_tag_counter.keys():
#         c_w, c_t = c_w_t
#         emission_dict[c_w_t] = word_tag_counter[c_w_t] / tag_counter[c_t]
#     for prev_tag in tag_tuples_counter.keys():
#         amount_prev = sum(tag_tuples_counter[prev_tag].values())
#         for current_tag, tuple_amount in tag_tuples_counter[prev_tag].items():
#             transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev
#     return transition_dict, emission_dict, tag_counter.keys(), word_counter.keys()
#
#
# def get_d_dictionaries(train_padded, test_set):
#     word_counter, tag_counter, tag_tuples_counter, word_tag_counter = dict(), dict(), dict(), dict()
#     transition_dict, emission_dict = dict(), dict()
#
#
#     for s in train_padded:
#         window_s = s[:-1]
#         for i, prev_tuple in enumerate(window_s):
#             prev_word, prev_tag = s[i]
#             current_word, current_tag = s[i + 1]
#
#             if prev_word not in word_counter.keys():
#                 word_counter[prev_word] = 0
#             word_counter[prev_word] += 1
#
#             if prev_tag not in tag_counter.keys():
#                 tag_counter[prev_tag] = 0
#             tag_counter[prev_tag] += 1
#
#             if prev_tuple not in word_tag_counter.keys():
#                 word_tag_counter[prev_tuple] = 0
#             word_tag_counter[prev_tuple] += 0
#
#             if (prev_tag, current_tag) not in tag_tuples_counter.keys():
#                 if prev_tag not in tag_tuples_counter.keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#                 elif current_tag not in tag_tuples_counter[prev_tag].keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#             tag_tuples_counter[prev_tag][current_tag] += 1
#
#     test_tuples = get_all_unknown_word_tag_dict(word_tag_counter, test_set)
#     for w_t in test_tuples:
#         word_tag_counter[w_t] = 0
#     for w_t in word_tag_counter.keys():
#         word_tag_counter[w_t] += 1
#
#     for c_w_t in word_tag_counter.keys():
#         c_w, c_t = c_w_t
#         emission_dict[c_w_t] = word_tag_counter[c_w_t] / (tag_counter.get(c_t, 0) + len(word_counter.keys()))
#     for prev_tag in tag_tuples_counter.keys():
#         amount_prev = sum(tag_tuples_counter[prev_tag].values())
#         for current_tag, tuple_amount in tag_tuples_counter[prev_tag].items():
#             transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev
#     return transition_dict, emission_dict, tag_counter.keys(), word_counter.keys()
#
#
# def solution_c(train_set, test_set):
#     train_padded, test_padded = add_padding_to_set(train_set, test_set)
#     transition_dict, emission_dict, train_tags, train_words = get_c_dictionaries(train_padded)
#
#     accuracy_known, known_count = 0, 0
#     accuracy_unknown, unknown_count = 0, 0
#     accuracy_total = 0
#
#     train_tags = list(train_tags)
#     train_tags.remove('START')
#
#     for i, sentence in enumerate(test_set):
#         words = [w_t[0] for w_t in sentence]
#         current_tags = viterbi(words, transition_dict, emission_dict, train_tags, train_words)
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
# def viterbi(sentence, transition_dict, emission_dict, train_tags, train_words):
#     m = len(sentence)  # x0,x1,.....,xn,x_(n+1)
#     t = len(train_tags)
#     pi, bp = np.zeros((m - 1, t)), np.zeros((m - 1, t))
#
#     # initial state
#     for i_t, tag in enumerate(train_tags):
#         pi[0, i_t] = 1
#         bp[0, i_t] = -1  # maybe change, maybe bug
#
#     # fill the table, without STOP
#     for k, word in enumerate(sentence[1:-1]):
#         for i_u, u_tag in enumerate(train_tags):
#             max_value, max_index = -1, -1
#             for i_w, w_tag in enumerate(train_tags):
#                 emission = get_emission(word, u_tag, train_words, emission_dict)
#                 transition = get_transition(w_tag, u_tag, transition_dict)
#                 prev_pi = pi[k - 1, i_w]
#                 value = emission * transition * prev_pi
#                 if value > max_value:
#                     max_value = value
#                     max_index = i_w
#             pi[k, i_u] = max_value
#             bp[k, i_u] = max_index
#
#     # get maximal value with STOP transition
#     max_tag_index, max_tag_value = -1, -1
#     for i_u, u_tag in enumerate(train_tags):
#         transition = get_transition(u_tag, 'STOP', transition_dict)
#         current_tag_value = pi[-1, i_u] * transition
#         if current_tag_value > max_tag_value:
#             max_tag_index = i_u
#             max_tag_value = current_tag_value
#
#     # trace back
#     predicted_tags = np.empty(m - 2, dtype=object)
#     curr_max_index = max_tag_index
#     predicted_tags[-1] = train_tags[max_tag_index]  # A[m-3]
#     for k in range(m - 4, -1, -1):
#         predicted_tags[k] = train_tags[int(bp[k + 1, curr_max_index])]
#         curr_max_index = int(bp[k + 1, curr_max_index])
#     return predicted_tags
#
#
# def get_error_rate(test_set, MLE_dict):
#     known_accuracy_counter = 0
#     unknown_accuracy_counter = 0
#     test_set_known_words_counter = 0
#     test_set_unknown_words_counter = 0
#     for s in test_set:
#         for w_t in s:
#             w, t = w_t
#             if w in MLE_dict.keys():
#                 if MLE_dict[w] == t:
#                     known_accuracy_counter += 1
#                 test_set_known_words_counter += 1
#             else:
#                 if t == 'NN':
#                     unknown_accuracy_counter += 1
#                 test_set_unknown_words_counter += 1
#     error_known = 1 - (known_accuracy_counter / test_set_known_words_counter)
#     error_unknown = 1 - (unknown_accuracy_counter / test_set_unknown_words_counter)
#     error_total = 1 - ((known_accuracy_counter + unknown_accuracy_counter) / (
#             test_set_known_words_counter + test_set_unknown_words_counter))
#     print("------------ QUESTION B ------------")
#     print("error for known words:", error_known)
#     print("------------------------------------")
#     print("error for unknown words:", error_unknown)
#     print("------------------------------------")
#     print("total error:", error_total)
#     print("------------------------------------")
#
#
# def get_MLE_dict(train_word_counter_dict, train_word_tag_counter_dict):
#     prob_dict = dict()
#     for w_t, count_w_t in train_word_tag_counter_dict.items():
#         w, t = w_t
#         if w not in prob_dict.keys():
#             prob_dict[w] = {t: count_w_t / train_word_counter_dict[w]}
#         else:
#             prob_dict[w][t] = count_w_t / train_word_counter_dict[w]
#     MLE_dict = dict()
#     for w in prob_dict.keys():
#         word_tags_dict = prob_dict[w]
#         MLE_dict[w] = max(word_tags_dict, key=word_tags_dict.get)
#     return MLE_dict
#
#
# def get_all_unknown_word_tag_dict(train_word_tag_dict, test_set):
#     unknown_word_tag_dict = []
#     known_word_tag_tuples = train_word_tag_dict.keys()
#     for s in test_set:
#         for w_t in s:
#             if w_t not in known_word_tag_tuples:
#                 unknown_word_tag_dict.append(w_t)
#     return unknown_word_tag_dict
#
# def solution_d(train_set, test_set):
#     train_padded, test_padded = add_padding_to_set(train_set, test_set)
#     transition_dict, emission_dict, train_tags, train_words = get_d_dictionaries(train_padded, test_set)
#
#     accuracy_known, known_count = 0, 0
#     accuracy_unknown, unknown_count = 0, 0
#     accuracy_total = 0
#
#     train_tags = list(train_tags)
#     train_tags.remove('START')
#
#     for i, sentence in enumerate(test_set):
#         words = [w_t[0] for w_t in sentence]
#         current_tags = viterbi(words, transition_dict, emission_dict, train_tags, train_words)
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
#     print("------------ QUESTION D ------------")
#     print("error for known words:", 1 - (accuracy_known / known_count))
#     print("------------------------------------")
#     print("error for unknown words:", 1 - (accuracy_unknown / unknown_count))
#     print("------------------------------------")
#     print("total error:", 1 - (accuracy_total / (known_count + unknown_count)))
#     print("------------------------------------")
#
#
# def get_pseudo_word(word,  count):
#     if count>5:
#         return word
#     if len(word) == 0:
#         return '__empty__'
#     if not word.isalpha():
#         return '__sign__'
#     if word.isdigit():
#         return '__number__'
#     if word[0].isupper():
#         return '__name__'
#     if len(word) > 2:
#         if word[-2:] == 'ed':
#             return '__past__'
#     if len(word) > 3:
#         if word[-3:] == 'ing':
#             return '__ing__'
#     if word[0] == "'"  and word[-1] == "'" or word[0] == '"' and word[-1] == '"':
#         return '__quote__'
#     return word
#
#
# def get_e_dictionaries(train_padded, test_padded):
#     word_counter, tag_counter, tag_tuples_counter, word_tag_counter = dict(), dict(), dict(), dict()
#     transition_dict, emission_dict = dict(), dict()
#
#     for s in train_padded:
#         window_s = s[:-1]
#         for i, prev_tuple in enumerate(window_s):
#             prev_word, prev_tag = s[i]
#             current_word, current_tag = s[i + 1]
#
#             if prev_word not in word_counter.keys():
#                 word_counter[prev_word] = 0
#             word_counter[prev_word] += 1
#
#             if prev_tag not in tag_counter.keys():
#                 tag_counter[prev_tag] = 0
#             tag_counter[prev_tag] += 1
#
#             if prev_tuple not in word_tag_counter.keys():
#                 word_tag_counter[prev_tuple] = 0
#             word_tag_counter[prev_tuple] += 0
#
#             if (prev_tag, current_tag) not in tag_tuples_counter.keys():
#                 if prev_tag not in tag_tuples_counter.keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#                 elif current_tag not in tag_tuples_counter[prev_tag].keys():
#                     tag_tuples_counter[prev_tag] = {current_tag: 0}
#             tag_tuples_counter[prev_tag][current_tag] += 1
#
#
#
#
#     for c_w_t in word_tag_counter.keys():
#         c_w, c_t = c_w_t
#         emission_dict[c_w_t] = word_tag_counter[c_w_t] / tag_counter[c_t]
#     for prev_tag in tag_tuples_counter.keys():
#         amount_prev = sum(tag_tuples_counter[prev_tag].values())
#         for current_tag, tuple_amount in tag_tuples_counter[prev_tag].items():
#             transition_dict[(prev_tag, current_tag)] = tuple_amount / amount_prev
#     return transition_dict, emission_dict, tag_counter.keys(), word_counter.keys()
#
# def solution_e(train_set, test_set):
#     train_padded, test_padded = add_padding_to_set(train_set, test_set)
#     transition_dict, emission_dict, train_tags, train_words = get_e_dictionaries(train_padded)
#
#     accuracy_known, known_count = 0, 0
#     accuracy_unknown, unknown_count = 0, 0
#     accuracy_total = 0
#
#     train_tags = list(train_tags)
#     train_tags.remove('START')
#
#     for i, sentence in enumerate(test_set):
#         words = [w_t[0] for w_t in sentence]
#         current_tags = viterbi(words, transition_dict, emission_dict, train_tags, train_words)
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
# if __name__ == '__main__':
#     train_set, test_set = solution_a()
#     # solution_b(train_set, test_set)
#     # solution_c(train_set, test_set)
#     solution_d(train_set, test_set)
