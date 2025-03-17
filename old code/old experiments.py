"""Experiment whether a subnetwork can be a subset of another"""

jtotal = []
jdirtytotal = []
for q in range(250):
    jcount = 0
    jdirtycount = 0
    n = 25
    c = 3
    G = Graphclass(15,10,2,True)
    k = len(G.full_subnetwork)
    dk = len(G.dirty_full_subnetwork)
    for k in range(len(G.full_subnetwork)):
        for i, subnetwork in enumerate(G.full_subnetwork[k]):
            sub = set(subnetwork)
            for j, othersubnetwork in enumerate(G.full_subnetwork[k]):
                if i != j:
                    other = set(othersubnetwork)
                    if other.issubset(sub):
                        jcount +=1
    for k in range(len(G.dirty_full_subnetwork)):
        for i, subnetwork in enumerate(G.dirty_full_subnetwork[k]):
            sub = set(subnetwork)
            for j, othersubnetwork in enumerate(G.dirty_full_subnetwork[k]):
                if i != j:
                    other = set(othersubnetwork)
                    if other.issubset(sub):
                        jdirtycount +=1
    jcount/=k
    jdirtycount/=dk
    jtotal.append(jcount)
    jdirtytotal.append(jdirtycount)
print('Average percent of subnetworks as subsets')
print(np.mean(jtotal))
print('Average percent of subnetworks as subsets in dirty subnetworks')
print(np.mean(jdirtytotal))




"""Experiment looking for evidences of cascading failure"""
for i in range(205):
    n = 15
    c = 2
    G = Graphclass(n,1,c,True)
    for node in G.zero_nodes:
        if node not in G.dirty_full_predicted:
            print('found one')
            for j in range(len(G.dirty_full_subnetwork)):
                for b in range(len(G.dirty_full_subnetwork[j])):
                    for v,_,_ in G.dirty_full_subnetwork[j][b]:
                        if v == node:
                            print(' dirty found node was in a subnetwork')
        if node not in G.full_predicted:
            print('found a hanging node which went to zero')
            for j in range(len(G.full_subnetwork)):
                for b in range(len(G.full_subnetwork[j])):
                    for v,_,_ in G.full_subnetwork[j][b]:
                        if v == node:
                            print('found node was in a subnetwork')
                            print(G.full_radii[j][b])
        print()

"""Experiment calculating how many graphs exhibit chaotic behavior"""
for i in range(25):
    chaos = 0
    n = np.random.randint(25,35)
    c = np.random.randint(3,8)
    G = Graphclass(15,10,2,True)
    e = G.initial_lyapunov()
    if e > 0:
        chaos +=1
chaos /= 100
print(f'percent chaotic {chaos}')


"""experiment to determine if it is possible for a node to be in a subnetwork with radius < 1 and not converge to zero"""

count = 0
none = True
while none:
    G = Graphclass(4,6,2, True)
    for node in G.dirty_full_predicted:
        if node not in G.zero_nodes:
            if G.converged[node] > 100:
               none = True 


count = 0
for i in range(10):
    G = Graphclass(10,25,3, True)
    for node in G.full_predicted:
        if node not in G.zero_nodes:
            print(G.converged[node])
            count +=1
print(count)

n = np.random.randint(3,15)
e = np.random.randint(2,5)
c = np.random.randint(2,6)
nf = True
while nf == True:
    G = Graphclass(n,n*e,c, True)
    for node in G.zero_nodes:
        if node not in G.dirty_full_predicted and len(G.dirty_full_predicted) == 0:
            nf = False
G.visualize_graph(G.G)
               
"""experiment to calculate the correctness of hueristics"""

def calculatecorrect(gt,test,n):
    correct = n
    for val in test:
        if val not in gt:
            correct -= 1
    for val in gt:
        if val not in test:
            correct -= 1
    return correct/n

for i in range(1,5):
    nvals = np.arange(4,25,2)
    avbfscorrect = []
    avswcorrect = []
    avgreedycorrect = []
    avfulltime = []
    avbfstime = []
    avswtime = []
    avgreedytime = []
    for n in nvals:
        print(n)
        if i <= 100:
            swcorrect = []

   
            timesw = []
      

            for _ in range(500):
                G = Graphclass(n,0,3, True)
    
                gt = G.zero_nodes
 
                G.swhueristic_predict(n/i)
 
                swvals = G.swhpredicted
     
           
                swcorrect.append(calculatecorrect(gt,swvals,n))
 
                timesw.append(G.swhflop)
   
            avswcorrect.append(np.mean(swcorrect))

            avswtime.append(np.mean(timesw))
    
    plt.figure()

    plt.plot(nvals, avswcorrect, label=f'SW n/{i}')

    plt.title('Average percent of nodes classified correctly')
    plt.ylabel('Percent correct')
    plt.xlabel('Number of nodes')
    plt.legend()
    plt.show()

    plt.figure()


    plt.plot(nvals, avswtime, label=f'SW n/{i}')
   
    plt.title('Average time')
    plt.ylabel('Time')
    plt.xlabel('Number of nodes')
    plt.legend()
    plt.show()


"""Comparisions of hueristics"""

colors = np.arange(2, 10,2)
# Initialize empty lists to hold the results for each color
all_avfulltimes = []
all_avtimes = []
all_avsub = []

# Loop over each color
for c in colors:
    nvals = np.arange(4, 32, 2)
    avfulltimes = []
    avtimes = []
    avsub = []

    # Loop over each value in nvals
    for n in nvals:
        print(f"Running for n={n}, color={c}")
        times = []
        fulltimes = []
        sub = []

      
        for i in range(250):
            G = Graphclass(n, 25, c, True)
            fulltimes.append(G.flop)
            times.append(G.flop / n)
            sub.append(len(G.full_subnetwork[0]))

        avfulltimes.append(np.mean(fulltimes))
        avtimes.append(np.mean(times))
        avsub.append(np.mean(sub))

    # Store the results for each color
    all_avfulltimes.append(avfulltimes)
    all_avtimes.append(avtimes)
    all_avsub.append(avsub)

# Plotting the results

# Plot for average computation time as a function of nodes
plt.figure()
for i, c in enumerate(colors):
    plt.plot(nvals, all_avtimes[i], label=f"{c} Colors")
plt.ylabel('Time in seconds')
plt.xlabel('Number of Nodes')
plt.title('Average computation time as a function of nodes with one target node')
plt.legend()
plt.show()

# Plot for time for full prediction as a function of nodes
plt.figure()
for i, c in enumerate(colors):
    plt.plot(nvals, all_avfulltimes[i], label=f"{c} Colors")
plt.ylabel('Time in seconds')
plt.xlabel('Number of nodes')
plt.title('Average time for full prediction as a function of nodes')
plt.legend()
plt.show()

# Plot for number of subnetworks as a function of nodes
plt.figure()
for i, c in enumerate(colors):
    plt.plot(nvals, all_avsub[i], label=f"{c} Colors")
plt.ylabel('Number of subnetworks')
plt.xlabel('Number of nodes')
plt.title('Average number of subnetworks as a function of nodes')
plt.legend()
plt.show()





for i in range(1,2):
    nvals = np.arange(4,12,2)
    avbfscorrect = []
    avswcorrect = []
    avgreedycorrect = []
    avfulltime = []
    avbfstime = []
    avswtime = []
    avgreedytime = []
    for n in nvals:
        print(n)
        if i <= 100:
            bfscorrect = []
            swcorrect = []
            greedycorrect = []
            timebfs = []
            timesw = []
            timegreedy = []
            timefull = []
            for _ in range(50):
                G = Graphclass(n,0,3, True)
                if G.converged:
                    gt = G.zero_nodes
                else:
                    gt = G.dirty_full_predicted
                G.swhueristic_predict(n/i)
                G.greedyhueristic_predict(n/i)
                bfsvals = G.dirty_predicted
                swvals = G.swhpredicted
                greedyvals = G.greedypredicted
                bfscorrect.append(calculatecorrect(gt,bfsvals,n))
                swcorrect.append(calculatecorrect(gt,swvals,n))
                greedycorrect.append(calculatecorrect(gt, greedyvals,n))
                timefull.append(G.fullflop)
                timebfs.append(G.bfsflop)
                timesw.append(G.swhflop)
                timegreedy.append(G.greedyflop)
            avbfscorrect.append(np.mean(bfscorrect))
            avswcorrect.append(np.mean(swcorrect))
            avgreedycorrect.append(np.mean(greedycorrect))
            avfulltime.append(np.mean(timefull))
            avbfstime.append(np.mean(timebfs))
            avswtime.append(np.mean(timesw))
            avgreedytime.append(np.mean(timegreedy))
    plt.figure()
    plt.plot(nvals, avbfscorrect, label="BFS")
    plt.plot(nvals, avswcorrect, label=f'SW n/{i}')
    plt.plot(nvals, avgreedycorrect, label =f'Greedy n/{i}')
    plt.title('Average percent of nodes classified correctly')
    plt.ylabel('Percent correct')
    plt.xlabel('Number of nodes')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(nvals, avfulltime, label='Full')
    plt.plot(nvals, avbfstime, label="BFS")
    plt.plot(nvals, avswtime, label=f'SW n/{i}')
    plt.plot(nvals, avgreedytime, label =f'Greedy n/{i}')
    plt.title('Average time')
    plt.ylabel('Time')
    plt.xlabel('Number of nodes')
    plt.legend()
    plt.show()


    """Condition numbers of subnetwork weight matrices"""

    n_trials = 1000  # Number of trials to run
condition_numbers = []

# Collect condition numbers
for _ in range(n_trials):
    G = Graphclass(20, 0.3, 3)
    condition_numbers.extend(G.condition)  # Append all condition numbers

# Convert to NumPy array for easier handling
condition_numbers = np.array(condition_numbers)

# Separate finite and infinite values
finite_values = condition_numbers[np.isfinite(condition_numbers)]
num_infs = np.sum(~np.isfinite(condition_numbers))  # Count infinite values

# Define histogram bins (logarithmic scale)
bins = np.logspace(np.log10(max(1, finite_values.min())), np.log10(finite_values.max()), 50)

# Compute histogram counts
hist_counts, bin_edges = np.histogram(finite_values, bins=bins)

# Normalize counts to percentages
total_subnetworks = len(condition_numbers)
hist_percentages = (hist_counts / total_subnetworks) * 100
inf_percentage = (num_infs / total_subnetworks) * 100  # Percentage of infinite values

# Plot histogram of finite values with log-scaled x-axis
plt.bar(bin_edges[:-1], hist_percentages, width=np.diff(bin_edges), align='edge', alpha=0.75, label='Finite Condition Numbers')

# Add a separate bar for infinite values at the rightmost position
if num_infs > 0:
    plt.bar(bin_edges[-1] * 1.5, inf_percentage, width=bin_edges[-1] * 0.5, color='red', label='Infinite Condition Numbers')

# Set x-axis to log scale
plt.xscale('log')

# Labeling and title
plt.xlabel('Condition Numbers of Subnetwork (Log Scale)')
plt.ylabel('Percentage of Subnetworks')
plt.title(f'Normalized Condition Numbers of Subnetworks in Graphs with 20 Nodes')
plt.legend()

# Show plot
plt.show()


n_trials = 10000  # Number of trials to run
condition_numbers = []

for _ in range(n_trials):
    G = Graphclass(20, 0.3, 3)
    for i in range(len(G.condition)):

        cond = G.condition[i]
        condition_numbers.append(cond)

# Plot histogram of the spectral condition numbers
condition_numbers = np.array(condition_numbers)

# Separate finite and infinite values
finite_values = condition_numbers[np.isfinite(condition_numbers)]
num_infs = np.sum(~np.isfinite(condition_numbers))  # Count infinite values

bins = np.logspace(np.log10(max(1, finite_values.min())), np.log10(finite_values.max()), 50)

# Plot histogram of finite values with log-scaled x-axis
plt.hist(finite_values, bins=bins, alpha=0.75, label='Finite Condition Numbers')

# Add a separate bar for infinite values at the rightmost position
if num_infs > 0:
    plt.bar(bins[-1] * 1.5, num_infs, width=bins[-1]*2, color='red', label='Infinite Condition Numbers')

# Set x-axis to log scale
plt.xscale('log')
plt.yscale('log')
# Labeling and title
plt.xlabel('Condition Numbers of Subnetwork (Log Scale)')
plt.ylabel('Frequency')
plt.title(f'Condition Numbers of Subnetworks in Graphs with 20 Nodes')
plt.legend()

# Show plot
plt.show()