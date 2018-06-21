# BullyProof
BullyProof is Cyberbully recognition and reporting software

## Introduction
BullyProof is a Google Chrome extension, built with a vision to stop cyberbully occuring in popular social media sites like, Twitter, Facebook, Instagram , etc. When installed as a Chrome Extension, it runs in background to check if you have been bullied or not. Then takes necessary actions.

## BullyProof - The Chrome Extension
BullyProof has two major components,
1. The timeline scrapper
2. A artificially intelligent model(AI Model) that reads through each of the scrapped media sessions(ex. A head tweet with all its replies corresponds to one media session) and predicts of the session qualifies as an instance of Cyberbully or not.<br>
The current version, is able to check for cyberbully in Twitter. We intend to extend this other social media as well.<br> 
The timeline scrapper uses the Twitter Tweepy API to scrape the timeline of a person who has logged in to BullyProof. The media sessions so obtained, are fed onto the AI Model to predict if its a bully or not.<br>
The number of Cyberbully instances in each week are taken into account and if they are found to be greater than certain threshold, it is reported to the “Guardian” (The Email of the guardian is taken when registering to BullyProof) through Email.

### Our Accomplishments
1. BullyProof can successfully scrape a person’s Twitter timeline and predict and instance of Cyberbully.
2. The AI Model that we have built can successfully distinguish between Cyberbullying and Cyber-Aggression.
3. Our Ai Model could give and Accuracy of 81% with and F1-Score of 81% when tested on a dataset borrowed from Carnegie Mellon University.
<br>
![alt picture](https://github.com/hlpr98/BullyProof/blob/master/Result.png)	
(Here class “-1” represents “Not Cyberbully” and “1” represents “Cyberbully”)

### Plans to improve this project
1. We want extend BullyProof to other social media sites.
2. Improve the UI of our app
3. Improve the F1-score of out AI Model
4. We want implement other ways of alerting people about Cyberbully.

### How to execute BullyProof

##### Pre-requisites
```
Python 3
Pickel
Scikit-learn
NLTK packages in python 3
Pandas
Numpy
Twitter Tweepy
```
#### Executing BullyProof:
* Add BullyProof extension(submitted) into Chrome
* Give the credential required.
* Check how the extension functions:
* Go to the source directory and execute tweepy.py and BullyProof.py in sequence.<br>
``` Use: Twitter handle used in the video(i.e victorjames663), which has bullying sessions already.```

<b>[Click here to watch the demonstration video on Youtube](https://www.youtube.com/watch?v=Sg1xTtknvd8)</b> 
