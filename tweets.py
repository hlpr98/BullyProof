#!/usr/bin/env python
# encoding: utf-8

import tweepy #https://github.com/tweepy/tweepy
import csv
import sys


#Twitter API credentials

consumer_key = 'CMde5m2cvi0FqRq7FGdcUMkbW'
consumer_secret = 'Tk0huoPBhqa7torgw5t9I7tDKBCF5XdGAL43b1hHzHaA2uUdWz'
access_key = "953993604015579136-zj0S88WcSbCcN2nQCYCxdmmrXa9Tk4g"
access_secret = "dVsJExjJWqdHBE4QX0iupvQgiCt7asCUJ9z22wOitG8Jk"


def get_all_tweets(name):

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth)

    #entry=[]
    replies=[] 
    non_bmp_map = dict.fromkeys(range(0x10000, 65536), 0xfffd)  
    toWrite = []
    count = 0
    for full_tweets in tweepy.Cursor(api.user_timeline,screen_name=name,timeout=999999).items(10):
      for tweet in tweepy.Cursor(api.search,q='to:'+name,result_type='recent',timeout=999999).items(1000):
        count2 = 0
        if hasattr(tweet, 'in_reply_to_status_id_str'):
          if (tweet.in_reply_to_status_id_str==full_tweets.id_str):
            replies.append(tweet.text)
      tmp = full_tweets.text.translate(non_bmp_map)
      count2 = count2 + 1
      time = tweet.created_at

      toWrite.append([])
      toWrite[count].append(time)
      toWrite[count].append(tmp)
      print("Tweet :",tmp)


      for elements in replies:
        count2 = count2 +1
        toWrite[count].append(elements)
        print("Replies :",elements) 
          
      for count2 in range(196):
        toWrite[count].append('empety')
      replies=[]
      count = count + 1


    file1 = open('tweets.csv','w')
    writer = csv.writer(file1)
    writer.writerows(toWrite)

     

    

if __name__ == '__main__':

    get_all_tweets("victorjames663")
