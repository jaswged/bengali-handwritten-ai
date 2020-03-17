# bengali-handwritten-ai
Repo for kaggle Bengali.Ai competition

Competition can be found here:
https://www.kaggle.com/c/bengaliai-cv19/overview

## Overview
Bengali is the 5th most spoken language with 100s of millions of speakers <br>
It has 49 letters with 11 vowels and 38 consonants <br>
It also has 18 potential diacritics or accents.<br>
This surmounts to over 13,000 different "graphemes" or the smallest unit in written language. English has about 250 for reference.<br>
It is written left to right with no "cases".
There is a vertical line at the top of each character that connects them together.

The challenge is to look at handwritten digits and correctly predict the root grapheme, vowel diacritic and consonant diacritic.

## To Use:
1. Download data from above link and extract it into the data directory.
2. Then run the train file to train and create the pytorch model.
3. To generate the test csv submission file run the CreateTestCsv.py file.

## Requirements
<ul>
    <li>Pytorch</li>
    <li>Pandas</li>
    <li>Numpy</li>
    <li>Tqdm</li>
    <li>Cv2</li>
</ul>

## About
The data is contained in parquet files, which are a columnar store where each pixel's greyscale value being its own column.
So each "row" is a separate image. 
White is 255 and black is 0.

## Results
I only found out about the competition with 5 days left, so I wsa not able to finish it in time to officially submit.

I did submit an attempt on the last day by forking one of the FastAi notebooks on kaggle to get some practice with submitting in the Kaggle environment.

<br/> I continued to work on this out of my own personal interest in deciphering languages and epanding my machine learning skills.