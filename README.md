# kagglefishCV
Solution for the old kaggle competition (https://www.kaggle.com/competitions/the-nature-conservancy-fisheries-monitoring/overview), that involves training of a representative ecnoder (BYOL + MobileNetv2) and creation of clusters in the embedding space

If I'd joined this comptetion 7 years ago and uploaded this solution I would have got a silver medal and top-100 of the leaderboard.
The data is super noisy, so I've split each image on smaller chunks using YOLO model. Then I trained the encoder using MobileNetv2 architecture and BYOL method of training ONLY on the training data (the test data is two times bigger, so I could have gotten a better represenation model if I used that data and trained for longer, but my 1660 isn't quite a match for this task). The next step was to create clusters out of that embeddings. I used OPTICS with cosine metric. Then on these pseudo-labels was trained KNN model with same (*as possible) parameters as OPTICS was. Using KNN I've created set of unique objects for each image. These sets were used to create logistic regression whose only purpose was to find the numbers of unique classes that correspond to fishes.

Options to upgrade the model:
* more train steps in encoder
* test data in encoder
* noisy student method (self distillation + pseudo-labels) on the very last step (probably just pseudo-labels would perform better)
