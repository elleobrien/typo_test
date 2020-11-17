# Typo test

Check how robust a sentiment classifier is to random typos in our dataset.

## Example
Given sentence A:

```I just love the new Honda I bought from Crystal.```

we can introduce a random typo to get sentence A':

```I jpst love the new Honda I bought from Crystal.```

Now we test whether our sentiment classifer S gives the same answer for S(A) and S(A'). If S(A) == S(A'), that's good! We can do this over hundreds or thousands of example sentences and estimate what percent of examples S(A) == S(A'). 

