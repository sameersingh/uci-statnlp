#!/bin/python
def read_twitter(dname="ner"):
    """Read the twitter train, dev, and test data from the default location.

    The returned object contains {train, test, dev}_{sents, labels}.
    """
    class Data: pass
    data = Data()
    # training data
    data.train_sents, data.train_labels = read_file("data/twitter_train." + dname)
    data.dev_sents, data.dev_labels = read_file("data/twitter_dev." + dname)
    # test data
    data.test_sents, data.test_labels = read_file("data/twitter_test." + dname)
    # print statistics
    print "Twitter %s data loaded." % dname
    print ".. # train sents", len(data.train_sents)
    print ".. # dev sents", len(data.dev_sents)
    print ".. # test sents", len(data.test_sents)
    return data

def read_file(filename):
    """Read the file in CONLL format, assumes one token and label per line."""
    sents = []
    labels = []
    with open(filename, 'r') as f:
        curr_sent = []
        curr_labels = []
        for line in f.readlines():
            if len(line.strip()) == 0:
                # sometimes there are empty sentences?
                if len(curr_sent) != 0:
                    # end of sentence
                    sents.append(curr_sent)
                    labels.append(curr_labels)
                    curr_sent = []
                    curr_labels = []
            else:
                token, label = line.split()
                curr_sent.append(unicode(token, 'utf-8'))
                curr_labels.append(label)
    return sents, labels

def write_preds(fname, sents, labels, preds):
    """Writes the output of a sentence in CONLL format, including predictions."""
    f = open(fname, "w")
    assert len(sents) == len(labels)
    assert len(sents) == len(preds)
    for i in xrange(len(sents)):
        write_sent(f, sents[i], labels[i], preds[i])
    f.close()

def write_sent(f, toks, labels, pred = None):
    """Writes the output of a sentence in CONLL format, including predictions (if pred is not None)"""
    for i in xrange(len(toks)):
        f.write(toks[i] + "\t" + labels[i])
        if pred is not None:
            f.write("\t" + pred[i])
        f.write("\n")
    f.write("\n")

def file_splitter(all_file, train_file, dev_file):
    """Splits the labeled data into train and dev, sentence-wise."""
    import random
    all_sents, all_labels = read_file(all_file)
    train_f = open(train_file, "w")
    dev_f = open(dev_file, "w")
    seed = 0
    dev_prop = 0.25
    rnd = random.Random(seed)
    for i in xrange(len(all_sents)):
        if rnd.random() < dev_prop:
            write_sent(dev_f, all_sents[i], all_labels[i])
        else:
            write_sent(train_f, all_sents[i], all_labels[i])
    train_f.close()
    dev_f.close()

def synthetic_data():
    """A very simple, three sentence dataset, that tests some generalization."""
    class Data: pass
    data = Data()
    data.train_sents = [
        [ "Obama", "is", "awesome" , "."],
        [ "Michelle", "is", "also", "awesome" , "."],
        [ "Awesome", "is", "Obama", "and", "Michelle", "."]
    ]
    data.train_labels = [
        [ "PER", "O", "ADJ" , "END"],
        [ "PER", "O", "O", "ADJ" , "END"],
        [ "ADJ", "O", "PER", "O", "PER", "END"]
    ]
    data.dev_sents = [
        [ "Michelle", "is", "awesome" , "."],
        [ "Obama", "is", "also", "awesome" , "."],
        [ "Good", "is", "Michelle", "and", "Obama", "."]
    ]
    data.dev_labels = [
        [ "PER", "O", "ADJ" , "END"],
        [ "PER", "O", "O", "ADJ" , "END"],
        [ "ADJ", "O", "PER", "O", "PER", "END"]
    ]
    data.test_sents = data.train_sents
    data.test_labels = data.train_labels
    return data

if __name__ == "__main__":
    # Do no run, the following function was used to generate the splits
    # file_splitter("data/twitter_train_all.pos", "data/twitter_train.pos", "data/twitter_dev.pos")

    dname = "pos" # or "ner"
    data = read_twitter(dname)
    # data = synthetic_data()

    import tagger
    tagger = tagger.LogisticRegressionTagger()
    #tagger = tagger.CRFPerceptron()

    # Train the tagger
    tagger.fit_data(data.train_sents, data.train_labels)

    # Evaluation (also writes out predictions)
    print "### Train evaluation"
    data.train_preds = tagger.evaluate_data(data.train_sents, data.train_labels)
    write_preds("data/twitter_train.%s.pred" % dname,
        data.train_sents, data.train_labels, data.train_preds)
    print "### Dev evaluation"
    data.dev_preds = tagger.evaluate_data(data.dev_sents, data.dev_labels)
    write_preds("data/twitter_dev.%s.pred" % dname, 
        data.dev_sents, data.dev_labels, data.dev_preds)
    print "### Test evaluation"
    data.test_preds = tagger.evaluate_data(data.test_sents, data.test_labels)
    write_preds("data/twitter_test.%s.pred" % dname, 
        data.test_sents, data.test_labels, data.test_preds)
    