import glob
import json
from keras.preprocessing.sequence import pad_sequences
import logging
import numpy as np
import os
import sys
import torch
from tqdm import trange, tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU {}!".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("No GPU :( using CPU")

# Global Variables
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
IMDB = os.path.join(DATA_DIR, 'aclImdb')
YELP = os.path.join(DATA_DIR, 'yelp')
r_state = 0


class ReviewDataset:
    """
    Class forsetting up and accessing train / test samples of imdb and yelp data

    IT REQUIRES THE DATA FOLDER TO BE SET UP APPROPRIATELY FIRST
    """
    def __init__(self, source="imdb", number_reviews=12500):       
        if source == "imdb":
            # IMDB Reviews
            self.positive_reviews_train = glob.glob(os.path.join(IMDB, 'train', 'pos', '*.txt'))[:number_reviews]
            self.negative_reviews_train = glob.glob(os.path.join(IMDB, 'train', 'neg', '*.txt'))[:number_reviews]

            self.positive_reviews_test = glob.glob(os.path.join(IMDB, 'test', 'pos', '*.txt'))[:number_reviews]
            self.negative_reviews_test = glob.glob(os.path.join(IMDB, 'test', 'neg', '*.txt'))[:number_reviews]

        elif source == "yelp":
            # Yelp Reviews
            sorted_glob_pos = sorted(glob.glob(os.path.join(YELP, 'pos', '*.txt')))
            sorted_glob_neg = sorted(glob.glob(os.path.join(YELP, 'neg', '*.txt')))

            self.positive_reviews_train = sorted_glob_pos[:number_reviews]
            self.negative_reviews_train = sorted_glob_neg[:number_reviews]

            self.positive_reviews_test = sorted_glob_pos[-number_reviews:]
            self.negative_reviews_test = sorted_glob_neg[-number_reviews:]

        else:
            raise Exception("Source {} is not available. Is it implemented?".format(source))


        assert(len(self.positive_reviews_test) > 0), "No files found. Have you set up your data folder properly?"
        assert(len(self.negative_reviews_test) > 0), "No files found. Have you set up your data folder properly?"
        assert(len(self.positive_reviews_train) > 0), "No files found. Have you set up your data folder properly?"
        assert(len(self.negative_reviews_train) > 0), "No files found. Have you set up your data folder properly?"

    @staticmethod
    def shrink_reviews(yelp_reviews_json, output_dir=DATA_DIR):
        """
        Static function for initialising the Yelp reviews. Assumes that the yelp tarball has been extracted
            
        yelp_reviews_json: Path to the large yelp review.json file
        output_dir: Highly recommended to leave as default
        
        
        """
        with open(yelp_reviews_json) as file:
            reader = file.readlines()
            os.makedirs(os.path.join(output_dir, 'pos'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'neg'), exist_ok=True)

            for i, review in enumerate(reader):
                review_dict = json.loads(review)

                # Discard short reviews and very long reviews
                length = len(review_dict['text'].split(' '))
                if length < 20 or length > 200:
                    continue

                # Yield neg label if <= 2 stars, pos label if >= 4 stars
                if review_dict['stars'] > 2 and review_dict['stars'] < 4:
                    continue
                elif review_dict['stars'] <= 2:
                    output = os.path.join(output_dir, 'neg', '{}.txt'.format(str(i)))
                elif review_dict['stars'] >= 4:
                    output = os.path.join(output_dir, 'pos', '{}.txt'.format(str(i)))

                with open(output, 'w') as writer:
                    writer.write(review_dict['text'])
    
    def reviewsAndLabels(self, test_train="train"):
        """
        Return the reviews and labels as numpy arrays

        test_train: "train", or "test", returning corresponding arrays
        """
        # Concatenate the files lists for positive and negative
        if test_train == "train":
            review_files = [*self.positive_reviews_train, *self.negative_reviews_train]
        elif test_train == "test":
            review_files = [*self.positive_reviews_test, *self.negative_reviews_test]
        else:
            raise Exception("test_train should  be 'test' or 'train, not {}".format(test_train))

        # Set up arrays for holding our sentences
        review_sentences = np.empty(len(review_files), dtype=object)

        # Populate sentence arrays
        for i, f in enumerate(review_files):
            with open(f) as review:
                text = review.read()
            review_sentences[i] = text
            
        # Instantiate labels as zeros
        review_labels = np.zeros(len(review_files))

        # Positive labeled reviews are 1
        no_pos = len(self.positive_reviews_train)
        review_labels[0:no_pos] = 1

        return review_sentences, review_labels

    @staticmethod
    def setUpData(reviews, labels, tokenizer, max_seq=256, split=0):
        """
        Takes an array of reviews, an array of labels and returns datasets for a given encoder

        reviews: np.array of reviews to that are the train data
        labels: np.array of labels of the reviews
        tokenizer: tokenizer (transformers library)
        max_seq: pad anything shorter, truncate anything longer
        split: will additionally return a test set of the corresponding split
        """
        encoded_data = [tokenizer.encode(sent, add_special_tokens=True, max_length=max_seq) for sent in reviews]
        encoded_data = pad_sequences(encoded_data, maxlen=max_seq, dtype="long",
                                    value=0, truncating="post", padding="post")
        pad_mask = np.where(encoded_data == 0, 0, 1)

        if split == 0:
            train_in, train_lab, train_mask = shuffle(encoded_data, labels, pad_mask, random_state=r_state)
            
            # Tensors only!
            train_in = torch.tensor(train_in, dtype=torch.long)
            train_lab = torch.tensor(train_lab, dtype=torch.long)
            train_mask = torch.tensor(train_mask, dtype=torch.long)
            train_data = TensorDataset(train_in, train_mask, train_lab)
            test_data = None
        
        elif split == "no_shuffle":
            train_in = torch.tensor(encoded_data, dtype=torch.long)
            train_lab = torch.tensor(labels, dtype=torch.long)
            train_mask = torch.tensor(pad_mask, dtype=torch.long)
            train_data = TensorDataset(train_in, train_mask, train_lab)
            test_data = None


        # Test split (could've used the defaults, but this gives more flexibility)
        else:
            train_in, test_in, train_lab, test_lab = train_test_split(encoded_data, labels,
                                                                    random_state=r_state, test_size=split)
            train_mask, test_mask, _, _ = train_test_split(pad_mask, labels,
                                                        random_state=r_state, test_size=split)

            # Tensors only please!
            train_in = torch.tensor(train_in, dtype=torch.long)
            test_in = torch.tensor(test_in, dtype=torch.long)
            train_lab = torch.tensor(train_lab, dtype=torch.long)
            test_lab = torch.tensor(test_lab, dtype=torch.long)
            train_mask = torch.tensor(train_mask, dtype=torch.long)
            test_mask = torch.tensor(test_mask, dtype=torch.long)

            # Set up the data for the model
            train_data = TensorDataset(train_in, train_mask, train_lab)
            test_data = TensorDataset(test_in, test_mask, test_lab)

        return train_data, test_data

def train(model, train_data, test_data, batch_size=8, epochs=2, lr=3e-5, adam_eps=1e-8):
    """
    Train loop for model. Adapted from https://mccormickml.com/2019/07/22/BERT-fine-tuning/
    and hugging faces GLUE transformer code and documentation

    model: pytorch model
    train_data: as produced by setUpData
    test_data: as produced by setUpData
    batch_size: for Bert uncased, recommend 8 for a T4 GPU
    epochs: number of epochs, recommend 2
    lr: learning rate, recommend
    """

    if device == torch.device("cuda"):
        model.cuda()
    # Set up objects for training
    losses = []

    sampler = RandomSampler(train_data)
    test_sampler = RandomSampler(test_data)

    loader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=lr, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=epochs * len(loader))

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_data))
    logger.info("  Num Epochs = %d", epochs)

    # Initialise losses
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    for epoch in trange(epochs, desc="Epoch"):
        epoch_iterator = tqdm(loader, desc="Iteration")
        model.train()
        for step, batch in enumerate(epoch_iterator):
            # Unpack the inputs from our dataloader
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Clear gradients
            model.zero_grad()

            # Forward pass
            outputs = model(b_input_ids, attention_mask=b_input_mask,
                            labels=b_labels)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            tr_loss += loss.item()  # accumulate losses

            # backwards
            loss.backward()

            # Clip gradients to avoid exploding
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Take a step
            optimizer.step()
            scheduler.step()

        # average loss and store
        avg_loss = tr_loss / len(loader)
        losses.append(avg_loss)
        logging.info("   Loss: {}".format(avg_loss))

        # Test Performance
        logger.info("***** Running Evaluation *****")

        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        
        # Evaluate data for one epoch
        for batch in test_loader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                outputs = model(b_input_ids,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences.
            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()
            tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

        # Log
        logger.info("  Accuracy: {}".format(eval_accuracy / nb_eval_steps))

        return losses, model


def evaluate(model, evaluation_data, batch_size=512, return_pred_labels=False, return_attentions=False):
    """
    Evaluation loop for pytorch model, returning batched accuracies. Adapted from
    https://mccormickml.com/2019/07/22/BERT-fine-tuning/ and hugging faces transformer code and documentation

    model: trained pytorch model
    evaluation_data: as produced by setUpData
    batch_size: recommend 512
    return_pred_labels: return the predictions as well as the accuracy
    return_attentions: return the attentions as well as the accuracy
    """
    nb_eval_steps = 0
    true_labels = []
    eval_accuracy = []
    pred_labels = []
    sms = []
    attentions = []
    sampler = SequentialSampler(evaluation_data)
    evaluation_data = DataLoader(evaluation_data, sampler=sampler, batch_size=batch_size)
    if device == torch.device("cuda"):
        model.cuda()

    model.eval()

    for batch in evaluation_data:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)

        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)

        # Get the "logits" output by the model. The "logits" are the output
        # values prior to applying an activation function like the softmax.
        logits = outputs[0]

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        if return_attentions:
            attention = outputs[-1]
            attention = attention.detach().cpu().numpy()
            attentions.append(attention)

        # Calculate the accuracy for this batch of test sentences.
        pred_flat = np.argmax(logits, axis=1).flatten()
        labels_flat = label_ids.flatten()
        tmp_eval_accuracy = np.sum(pred_flat == labels_flat) / len(labels_flat)


        # Return predicted labels
        if return_pred_labels:
            pred_labels.append(pred_flat)
            true_labels.append(labels_flat)
            sm = np.exp(logits.T) / np.sum(np.exp(logits), axis=1)
            sms.append(sm.T)


            

        # Accumulate the total accuracy.
        eval_accuracy.append(tmp_eval_accuracy)

        # Track the number of batches
        nb_eval_steps += 1

    return eval_accuracy, pred_labels, sms, attentions, true_labels

