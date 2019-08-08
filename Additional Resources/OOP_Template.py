# Import the necessary methods from tweepy library


# call config file
# app_config = get_app_config()


class MyClass(object):
    def __init__(self, new_run=True):

        # Some variables that we don't need to save can just be variables..
        access_token = app_config["twitter_dev_credentials"]["access_token"]
        access_token_secret = app_config["twitter_dev_credentials"]["access_token_secret"]
        consumer_key = app_config["twitter_dev_credentials"]["consumer_key"]
        consumer_secret = app_config["twitter_dev_credentials"]["consumer_secret"]

        # Other variables we want to store in self
        self.pickle_path = app_config["app"]["pickle_path"]
        if new_run:
            self.tweets = []
        else:
            print("Loading in Current Pickle")
            with open(self.pickle_path, 'rb') as f:
                self.tweets = pickle.load(f)


    def another_function(self, regular_job=True, return_tweets=True):



    def run(self, test_run=False):
        """ This Method will call other methods. It itself will be called by main. """


if __name__ == '__main__':
    print("creating tweet extractor object")
    myObject = MyClass(new_run=True)
    print("running extractor object")
    myObject.run()

