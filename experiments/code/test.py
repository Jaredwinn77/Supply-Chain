from Graphclass_module import Graphclass
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

def count(n, e, c):
    count = 0
    for _ in range(100):
        found = False
        G = Graphclass(n, e, c, 'random', True)
        # Debugging output to check graph conditions
        if len(G.zero_nodes) != 0 and len(G.full_predicted) == 0:
            found = True
        if found:
            count += 1
    return count




nvals = np.arange(2,8,1)
cvals = np.arange(1,5,1)
print('yippe')
n_list, e_list, c_list, output_list = [], [], [], []
for n in nvals:
    for e in np.arange(n+2,n*3,1):  # Ensure e >= n by starting e from n
        for c in cvals:
            # Compute the output for each combination
            output = count(int(n), int(e), int(c))  # Call count with integer values
            n_list.append(n)
            e_list.append(e)
            c_list.append(c)
            output_list.append(output)
    print(n)
# Convert lists to numpy arrays for plotting
n_list = np.array(n_list)
e_list = np.array(e_list)
c_list = np.array(c_list)
output_list = np.array(output_list)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting the 3D scatter plot
sc = ax.scatter(n_list, e_list, c_list, c=output_list, cmap='hot', alpha =.7)

# Add labels and color bar
ax.set_xlabel('Number of Nodes')
ax.set_ylabel('Number of Edges')
ax.set_zlabel('Number of Colors')
ax.view_init(elev=30, azim=45)
#plt.colorbar(sc, label='Number of abnormal graphs')
ax.set_xticks(np.arange(min(n_list), max(n_list) + 1,2))
ax.set_yticks(np.arange(min(e_list), max(e_list) + 1,5))
ax.set_zticks(np.arange(min(c_list), max(c_list) + 1))
plt.title('Abundance of abnormal zero graphs')
# Show the plot
with open('num_all_zero.pkl', 'wb') as f:
    pickle.dump(fig, f)
plt.show()