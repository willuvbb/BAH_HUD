Thanks for looking at my WebApp!

To run, you'll need to set up a few things first. 

First thing, and it's a pain but not too bad, is making 
an elasticsearch index. "Ahh, what's that!?" That's what 
I said at first, and it's a good thing I had Tommy Englert 
to help me out. You should probably contact him if you want 
actual help. But, you can figure it out. Go to this link:
https://www.elastic.co/downloads/elasticsearch

Literally just follow the instructions on that page. Download 
it, then follow the 4 steps.

You have to go and run the "bin/elasticsearch" from your 
terminal every time that you want to run the webapp. That's 
where the tweets are stored and loaded from.

Then, run "setup_elastic.py". This will store the data which 
is in this folder in .csv files into elasticsearch indexes.

Then, you can run the webapp. Just run "app.py" in an IDE 
or type "python app.py" in the terminal, making sure that 
you're in the right folder. 

To edit the "content" of the app, go to templates/Dashboard.html, 
and work your magic.

Ask me if you have any questions.

-Will
willuvbb@gmail.com
