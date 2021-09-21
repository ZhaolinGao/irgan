import tensorflow as tf
from dis_model_dns import DIS
import numpy as np
import multiprocessing
import scipy.sparse as sp

def parse_args():
    parser = argparse.ArgumentParser(description="Run IRGAN.")
    parser.add_argument('--emb_dim', type=int, default=64,
                        help='embedding dimension')
    parser.add_argument('--num_neg', type=int, default=10,
                        help='number of negatives')
    parser.add_argument('--dataset', type=str, default="TAFA-digital-music",
                        help='dataset')
    return parser.parse_args()

def generate_dns(sess, model, num_neg, all_items):
    data = []
    for u in user_pos_train:
        pos = user_pos_train[u]
        all_rating = sess.run(model.dns_rating, {model.u: u})
        all_rating = np.array(all_rating)
        neg = []
        candidates = list(all_items - set(pos))

        for _ in range(len(pos)):
            choice = np.random.choice(candidates, num_neg)
            choice_score = all_rating[choice]
            neg.append(choice[np.argmax(choice_score)])

        for i in range(len(pos)):
            data.append([u, pos[i], neg[i]])

    return data


def ndcg_func(ground_truths, ranks):
    result = 0
    for i, (rank, ground_truth) in enumerate(zip(ranks, ground_truths)):
        len_rank = len(rank)
        len_gt = len(ground_truth)
        idcg_len = min(len_gt, len_rank)

        # calculate idcg
        idcg = np.cumsum(1.0 / np.log2(np.arange(2, len_rank + 2)))
        idcg[idcg_len:] = idcg[idcg_len-1]

        dcg = np.cumsum([1.0/np.log2(idx+2) if item in ground_truth else 0.0 for idx, item in enumerate(rank)])
        result += dcg / idcg
    return result / len(ranks)

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(actual)
    true_users = 0
    for i, v in actual.items():
        act_set = set(v)
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    assert num_users == true_users
    return sum_recall / true_users

def eval(model, sess, train_data, test_data, num_user, num_item):
    user_batch = list(range(num_user))
    predictions = sess.run(model.all_rating, {model.u: user_batch})

    topk = 20
    predictions[train_data.nonzero()] = np.NINF

    ind = np.argpartition(predictions, -topk)
    ind = ind[:, -topk:]
    arr_ind = predictions[np.arange(len(predictions))[:, None], ind]
    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(predictions)), ::-1]
    pred_list = ind[np.arange(len(predictions))[:, None], arr_ind_argsort]

    recall = []
    for k in [5, 10, 20]:
        recall.append(recall_at_k(test_data, pred_list, k))

    all_ndcg = ndcg_func([*test_data.values()], pred_list[list(test_data.keys())])
    ndcg = [all_ndcg[x-1] for x in [5, 10, 20]]

    return recall, ndcg


# def dcg_at_k(r, k):
#     r = np.asfarray(r)[:k]
#     return np.sum(r / np.log2(np.arange(2, r.size + 2)))


# def ndcg_at_k(r, k):
#     dcg_max = dcg_at_k(sorted(r, reverse=True), k)
#     if not dcg_max:
#         return 0.
#     return dcg_at_k(r, k) / dcg_max


# def simple_test_one_user(x):
#     rating = x[0]
#     u = x[1]

#     test_items = list(all_items - set(user_pos_train[u]))
#     item_score = []
#     for i in test_items:
#         item_score.append((i, rating[i]))

#     item_score = sorted(item_score, key=lambda x: x[1], reverse=True)
#     item_sort = [x[0] for x in item_score]

#     r = []
#     for i in item_sort:
#         if i in user_pos_test[u]:
#             r.append(1)
#         else:
#             r.append(0)

#     p_3 = np.mean(r[:3])
#     p_5 = np.mean(r[:5])
#     p_10 = np.mean(r[:10])

#     ndcg_3 = ndcg_at_k(r, 3)
#     ndcg_5 = ndcg_at_k(r, 5)
#     ndcg_10 = ndcg_at_k(r, 10)

#     return np.array([p_3, p_5, p_10, ndcg_3, ndcg_5, ndcg_10])


# def simple_test(sess, model):
#     result = np.array([0.] * 6)
#     pool = multiprocessing.Pool(cores)
#     batch_size = 128
#     test_users = list(user_pos_test.keys())
#     test_user_num = len(test_users)
#     index = 0
#     while True:
#         if index >= test_user_num:
#             break
#         user_batch = test_users[index:index + batch_size]
#         index += batch_size

#         user_batch_rating = sess.run(model.all_rating, {model.u: user_batch})
#         user_batch_rating_uid = zip(user_batch_rating, user_batch)
#         batch_result = pool.map(simple_test_one_user, user_batch_rating_uid)
#         for re in batch_result:
#             result += re

#     pool.close()
#     ret = result / test_user_num
#     ret = list(ret)
#     return ret


def binarize_dataset(threshold, training_users, training_items, training_ratings):
    for i in range(len(training_ratings)):
        if training_ratings[i] > threshold:
            training_ratings[i] = 1
        else:
            training_ratings[i] = 0
    training_users = [training_users[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_items = [training_items[i] for i in range(len(training_ratings)) if training_ratings[i] != 0]
    training_ratings = [rating for rating in training_ratings if rating != 0]
    return training_users, training_items, training_ratings


# def generate_uniform(filename):
#     data = []
#     print('uniform negative sampling...')
#     for u in user_pos_train:
#         pos = user_pos_train[u]
#         candidates = list(all_items - set(pos))
#         neg = np.random.choice(candidates, len(pos))
#         pos = np.array(pos)

#         for i in range(len(pos)):
#             data.append(str(u) + '\t' + str(pos[i]) + '\t' + str(neg[i]))

#     with open(filename, 'w')as fout:
#         fout.write('\n'.join(data))


def main():
    cores = multiprocessing.cpu_count()
    args = parse_args()
    workdir = "./dataset/"+args.dataset

    if args.dataset in ['TAFA-digital-music', 'TAFA-cd', 'TAFA-grocery']:
        train = pickle.load(open('./dataset/'+args.dataset+'/train.pkl', "rb"))
        train_users, train_items, train_ratings = train
        train_users, train_items, train_ratings = binarize_dataset(3, train_users, train_items, train_ratings)
        train_data = []
        for uid, iid in zip(train_users, train_items):
            train_data.append([uid, iid])

        test = pickle.load(open('./dataset/'+args.dataset+'/val.pkl', "rb"))
        test_users, test_items, test_ratings = test
        test_users, test_items, test_ratings = binarize_dataset(3, test_users, test_items, test_ratings)
        test_data = []
        for uid, iid in zip(test_users, test_items):
            test_data.append([uid, iid])

    elif args.dataset in ['amazon-book20', 'amazon-cd', 'yelp4']:
        train = pickle.load(open('./dataset/'+args.dataset+'/train.pkl', "rb"))
        train_data = []
        for user, items in train.items():
            for item in items:
                train_data.append([user, item])

        test = pickle.load(open('./dataset/'+args.dataset+'/test.pkl', "rb"))
        test_data = []
        for user, items in test.items():
            for item in items:
                test_data.append([user, item])

    train_data = np.array(train_data)
    test_data = np.array(test_data)
    num_user = max(np.max(train_data[:, 0]), np.max(test_data[:, 0])) + 1
    num_item = max(np.max(train_data[:, 1]), np.max(test_data[:, 1])) + 1
    all_items = set(range(num_item))

    user_pos_train = {}
    user_pos_test = {}
    train_mat = sp.dok_matrix((num_user, num_item), dtype=np.float32)
    for i in range(train_data.shape[0]):
        train_mat[train_data[i, 0], train_data[i, 1]] = 1.0
        if train_data[i, 0] in user_pos_train:
            user_pos_train[train_data[i, 0]].append(train_data[i, 1])
        else:
            user_pos_train[train_data[i, 0]] = [train_data[i, 1]]
    for i in range(test_data.shape[0]):
        if test_data[i, 0] in user_pos_test:
            user_pos_test[test_data[i, 0]].append(test_data[i, 1])
        else:
            user_pos_test[test_data[i, 0]] = [test_data[i, 1]]

    DIS_MODEL_FILE = workdir + "model_dns.pkl"

    np.random.seed(70)
    param = None
    discriminator = DIS(num_item, num_user, args.emb_dim, lamda=0.1, param=param, initdelta=0.05, learning_rate=0.05)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    dis_log = open(workdir + 'dis_log_dns.txt', 'wb')
    print("dis ", simple_test(sess, discriminator))
    best_p5 = 0.

    # generate_uniform(DIS_TRAIN_FILE) # Uniformly sample negative examples

    for epoch in range(80):
        data = generate_dns(sess, discriminator, args.num_neg, all_items)  # dynamic negative sample
        for line in data:
            u = line[0]
            i = line[1]
            j = line[2]
            _ = sess.run(discriminator.d_updates,
                         feed_dict={discriminator.u: [u], discriminator.pos: [i],
                                    discriminator.neg: [j]})

        result = simple_test(sess, discriminator, train_mat, user_pos_test, num_user, num_item)
        print("epoch ", epoch, "dis: ", result)
        if sum(result) > sum(best_p5):
            best_p5 = result
            discriminator.save_model(sess, DIS_MODEL_FILE)

        buf = '\t'.join([str(x) for x in result])
        dis_log.write(str(epoch) + '\t' + buf + '\n')
        dis_log.flush()

    dis_log.close()
    print("best results: best_p5")


if __name__ == '__main__':
    main()
