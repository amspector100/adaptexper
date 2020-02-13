# Observations/To Chat About

0. What's the right formulation of SDP?

(a) One that works: minimize l_pth norm of (Sigma_{ij} - S_{ij}/Sigma_{ij})
but this is very slow.

(b) sum(abs(Sigma - S)) works pretty well

(c) What do we care about here? Intuitively what gives good power? It's no longer 0 corr?

1. 

2. In correlated settings, is rather conservative - why?
Does this have to do w/ the supermartingale bound? (IDTS)
3. Am having trouble finding good group_lasso package
but found a promising new lead this morning

## How to decide if two groups are "different"

1. Look at something like the canonical correlation
ccor(X, tilde(X)) = max_{u \in span(X), v \in span\tilde(X)} empiricalcorr(u, v)
2. Do we want to weight canonical correlations?
Conceptually similar to how to pick the weightings in the group lasso?




# Evaluating group knockoffs

## Debugging

Be on the lookout for indexing errors for the groups

## Look at regular power

I.e. 
$$\sum_{j \in \hat{S}, j : X_{G_j} \Perp X_{-G_j}}$$


Group_lasso - try using https://glm-tools.github.io/pyglmnet/auto_examples/plot_group_lasso.html

## For SDP solvers

- Try using frobenius norm, linear minimization/maximization
	- What is equicorrelated equivalent to?
	- Can implemenet ASDP 

# For constructing graphs
 

## Sample Splitting

## Schur complement stuff

## Cutoffs

## Averages over settings


# For discussion

1. How to make canonical correlation convex? Convex-ify?
2. ASDP approximation for speedup?
3. Access to computing clusters?

# IMPORTANT NOTE on the Oracle

- The oracle should take expectations over X's, 
meaning that you should sample X from the **same**
precision/covariance matrix a bunch of times.
- Nonsplit power may then be higher than oracle power,
which would imply "double-dipping". This means that
we should NOT take the expectation over the X's in this case.
- No real need to average over the AR(5,1) graphs
- TRY RUNNING THIS WITH A LOT LESS COARSENESS


Use frobenius norm for now, come back to this later if it seems 
unhappy.

# Contracts with Tejal

(1) I'm long at 0.55 for Ben/Hemanth, 10 contracts at $1
(2) I'm short at 0.95 for Ed Bracey, 10 contracts at $1


## Implementation Notes
- Need to be able to submit S as a parameter to knockoff generator

# ASDP Approximation

- Best grouping we choose will depend on $n$
- Can use slightly bigger blocks than your blocks
- Don't scale up to experimentation phase

# Experimentation phase

- Adding a new parameter for ASDP
- Decrease precision of convex optimization (doesn't
need to control within 4th decimal place, instead of 1st)

# Tuning parameters of Graphical Lasso
- Play with different clustering methods (Frobenius)
- Pick the right statistics
- Play with different Y|X models 
(logistic, play with sparsity, magnitude of coefficients)
- Cluster access 

Would be cool if we could prove something about that method,
because it doesn't seem that far-fetched. 

Proof method: power curve is fairly deterministic,
with big enough n (maybe p matters too), you can
get a deterministic power curve. 

I also want things to somehow be sharply peaked
so you always make the same decision over random X.
First dip of the data doesn't rely on the data. Extreme
reduction of the data: we're only looking at this
one curve caused by the data, and we're only looking
at the 

Also want to prove that our statistic is smooth in this
choice.

# Consider multidimensional trees 

Put it in a tree because it's satisfying computationally,
we could consider more complicated ways of doing it.

Doing theoretical work, that's an interesting question to ask.

Right measure: how many possible partitions are there.
Here, the answer is at most p. Maybe that's the fundamental number:
we'll need n * p > p.


Artificially coarsen to get rid of it. 

I have $k$ possible partitions, there is an optimal partition.
All you need to show: as my data gets large, with high prob,
I always pick the same partition. Bound your error
by the probability you don't pick that partition. 

Probability that the max of the random curve you draw
is equal to the nonrandom curve you draw. You can talk about that

For a fixed, finite number of partitions, you know the limit
will be the smallest partition. 

# Programming Thoughts

1. Make sure you 

# Cluster thoguhts
1. Right place
1.5 Where to store, how to move
Can I git clone in the batch script? 
2. Best way to log, can I just print?
3. Paralellization tricks
4. 

ASDP
resuse

# Other benchmarks to test sample reuse against
1. Test against an oracle which 
gets ideal partitions for n/2 
and then both (i) it recycles the other
half of the data and (ii) it does not
2. p = 300, ErdosRenyi graph
3. Optimal cutoff logarithmic in n
Doubling n actually barely changes 
the level that you want. In simple
cases it should look like: scaling
should look like sqrt(n).


What grouping is good given n

# How to combine groups

Define levels by the average height
of each cluster, average correlation,
etc. 

Do both: 

# Scale up: answer questions we've
# really been wondering which have been
# limited by p. Same questions 
# Big enough p that makes a difference

# Plots: line plots with something (n, p)
# on the x axis, power on the y axis,
# and a few different curves for different
# methods. (FDR plot as well to check
# things are working)

- X axis: varies n, let p be 1000
(Non-group SDP, 1000 is super easy)
- Take advantage of 32 cores on single 
cluster
- One time cost against many many simulations,
wouldn't bat an eye if it took a few
days to run.

256 gigs of ram, 32 core per node
8 nodes total
janson big-mem partition: one more node w/ 32 cores, 1 T of ram

Other things to keep track of:
	- What oracle cutoffs are (compare them)
	- See what double-dipping one finds,
	what non-double-dipping one finds,
	etc. 
	- Would like to know if the oracle has
	a correlation cutoff, distribution over
	group sizes, what is the distribution over all of those things for the two methods we have 

Figure out how sample-split compares to 
oracle

Try to break double-dipping if you can

Explore other oracles so we understand
why sample-reuse seems to be doing so well

Small example of cluster use as well.

# Notes
1. Confirmed that the S matrix generated is truly independent of n

# Tonight
1. Caching of previous runs
2. Master plotting function
3. Pull stuff onto local computer

# Cluster size doubled

janson_cascade has 288 nodes and if you submit to them
newer, faster, cores (worth more than other cores)
3 blades of 2 nodes of 48 cores

# ASDP performance

1. ErdosRenyi is not very clustery, 
social networks follows power law for example

Not really sure apriori how to define those clusters,
but you would still expect to be able to interpret
a friend group.

You might actually combine these two things: I'm 
only going to hierarchically cluster them if they're
also friends, within one separation.

Run something quite large with q = 0.1

# SDP 

KNOCKOFF ZOOM

2. SDP is not trivially parallelizable, so 
should use different cores.
Parallelize as much as possible to the extent that
paralleilization helps. (That works better
when running the cluster at capacity).

Look at cluster usage, see what happens.
Feel comfortable emailing me: "I urgently need
to run this."

Analogue: |E - \theta| < some bound 
Instead: Utility (expected power) < some bound

# SuSie

1. At some point pull their code,
start comparing our methods to theirs
2. Expected: their method will take
forever, and also, ours will
perform better for more effects.

# Pre-thanksgiving meeting

To talk about: see folder

Other papers:
1. https://arxiv.org/pdf/1503.00334.pdf (2015 Reid/Tibirshani paper)
2. http://users.cms.caltech.edu/~hou/papers/Prototype-knockoff-2019.pdf (Hou from harvard stat dept)

Then on diff privacy:
(1) Method is definitely applicable: we may have to replace 'diff privacy' with martingale analysis
(bc what is diff privacy for n = 1? although we could try for n = 1?)
Or is there a smart way to do this?
(2) What are we trying to estmate?

3. SUMMER STUFF - you'll regret it if you don't ask about this


# Over thanksgiving
1. Change to make more object oriented
2. Implement some competitor methods


Try eoracle: see what happens when you pick max empirical power

# To discuss:
1. 

# Notes from Alec's Presentation

1. Maintainability > performance
You can always spend time optimizing later

2. Only fix bottlenecks afer you find them

3. Always use Pylint

4. Functions should be used when you want to abstract repeated functionality

5. Classes should be employed to combine related functions with relevant state information and data

6. When to commit code and how often. Better to do it on the smaller end of things.
Make like 50 lines of code, that's worth committing. 



## Unit tests
0. Build failsafes in the actual code (asserts, value errors, etc)
so that things faily quickly!

1. Write tests prior to writing the actual code

2. Testing randomly is good bc it may tests cases you weren't
expecting

3. Test oracle (inverse function): figure out what your function
is supposed to return given a particular set of inputs

4. How to test random functions - have a random component 
"swapped out" or "mocked up," which has the same functionality
and inputs/outputs but a more naive/simple version. Remove
the randomness from your function upon request.

## Memory Tracking

1. If you want to track memory, maybe use Guppy or Memory Profiler

# More bug hunting

1. Try rewriting the experiment base 
2. Replace any row of the knockoffs with an exact copy of X
3. Share knockoffs
4. Hope they're aren't weird, pathological edge cases where lasso
totally breaks for a fairly reasonable knockoff generation mechanism

# To do

1. Bug hunt for LASSO
2. Complete the proof
3. Martingales
4. Counting arguments

On 1: Differences between msesia's code and mine:
(a) applying his feature fn to my knockoffs --> more power/rejections