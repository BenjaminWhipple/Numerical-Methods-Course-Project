import numpy as np
import matplotlib.pyplot as plt
import time
import dill

# System parameters
Lx, Ly = 30,30

t0 = 0
tf = 16.0

time_steps = int((tf-t0)*25)

t_eval = np.linspace(t0,tf,time_steps)

times1 = [0,5*25,10*25,16*25-1] # Full time
times2 = [0,2*25,4*25,6*25] # Partial time
times3 = [0,1*25,2*25,3*25]
#### Considering central infection site.

with open("Intermediate-Objects/solution.pkl", "rb") as f:
    sol1 = dill.load(f).cpu().numpy()

print(sol1)
print(sol1.shape)

NUM_STATES = 6

T_sol,I_sol,I_star_sol,R_sol,V_sol,F_sol = [sol1[:,i,:,:] for i in range(NUM_STATES)]

## Virions only

# Considering full time
fig, ax = plt.subplots(1, 4, figsize=(16, 4))


epsilon = 1e-8
V_datasets = [V_sol[i]+epsilon for i in times1]
V_min = max(min([V.min() for V in V_datasets]),0)
V_max = max([V.max() for V in V_datasets])

print(V_min)
print(V_max)


for i, t in enumerate(times1):
    im = ax[i].imshow(V_sol[t]+epsilon, extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    ax[i].set_title(f"t={t_eval[t]:.1f} (days)")
    ax[i].set_xlabel("x")
    
ax[0].set_ylabel("y")
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"Virion Quantity",rotation=270,labelpad=20.0)
plt.savefig("Images/virions1.png")

# Considering partial time.
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

epsilon = 1e-8
V_datasets = [(V_sol[i]+epsilon) for i in times2]
V_min = min([V.min() for V in V_datasets])
V_max = max([V.max() for V in V_datasets])

for i, t in enumerate(times2):
    im = ax[i].imshow((V_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    ax[i].set_title(f"t={t_eval[t]:.1f} (days)")
    ax[i].set_xlabel("x")
    
ax[0].set_ylabel("y")
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\log_{10}$ Virion Quantity",rotation=270,labelpad=20.0)
plt.savefig("Images/virions2.png")


## Virions and IFN
# Partial Time
fig, ax = plt.subplots(2, 4, figsize=(10, 4))
times3 = [0,1*25,2*25,3*25]

epsilon = 1e-8
V_datasets = [np.log10(V_sol[i]+epsilon) for i in times3] + [np.log10(F_sol[i]+epsilon) for i in times3]
V_min = min([V.min() for V in V_datasets])
V_max = max([V.max() for V in V_datasets])
ims = []

# Not DRY code, but quick.
for i, t in enumerate(times3):
    im = ax[0,i].imshow(np.log10(V_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
    	ax[0,i].set_yticklabels([])

    ax[0,i].set_xticklabels([])

    ax[0,i].set_title(f"t={t_eval[t]:.1f} (days)")
    ims.append(im)

for i, t in enumerate(times3):
    im = ax[1,i].imshow(np.log10(F_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
    	ax[1,i].set_yticklabels([])

    ims.append(im)

fig.text(0.07, 0.7, "Virions", ha="center", va="center", fontsize=12, rotation=90)
fig.text(0.07, 0.3, "Interferons", ha="center", va="center", fontsize=12, rotation=90)

cbar = fig.colorbar(ims[0], ax=ax[:,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\log_{10}$ Quantity",rotation=270,labelpad=20.0)
#plt.tight_layout()
plt.savefig("Images/virions_and_interferons.png")


## Cell Dynamics
T_sol,I_sol,I_star_sol,R_sol
V_datasets = [np.log10(T_sol[i]+epsilon) for i in times2] + [np.log10(I_sol[i]+epsilon) for i in times2] + [np.log10(I_star_sol[i]+epsilon) for i in times2] + [np.log10(R_sol[i]+epsilon) for i in times2]
V_min = min([V.min() for V in V_datasets])
V_max = max([V.max() for V in V_datasets])
ims = []

fig, ax = plt.subplots(4, 4, figsize=(10, 10),squeeze=True)
# Not DRY code, but quick.
for i, t in enumerate(times3):
    im = ax[0,i].imshow(np.log10(T_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[0,i].set_yticklabels([])

    ax[0,i].set_xticklabels([])

    ax[0,i].set_title(f"t={t_eval[t]:.1f} (days)")
    ims.append(im)

for i, t in enumerate(times3):
    im = ax[1,i].imshow(np.log10(I_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[1,i].set_yticklabels([])

    ims.append(im)

for i, t in enumerate(times3):
    im = ax[2,i].imshow(np.log10(I_star_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[2,i].set_yticklabels([])

    ims.append(im)

for i, t in enumerate(times3):
    im = ax[3,i].imshow(np.log10(R_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[3,i].set_yticklabels([])

    ims.append(im)

fig.text(0.03, 0.8, r"Uninfected ($T$)", ha="center", va="center", fontsize=10)#, rotation=90)
fig.text(0.03, 0.6, r"Producing ($I$)", ha="center", va="center", fontsize=10)
fig.text(0.03, 0.4, r"Antiviral ($I^{*}$)", ha="center", va="center", fontsize=10)
fig.text(0.03, 0.2, r"Refractory ($R$)", ha="center", va="center", fontsize=10)

cbar = fig.colorbar(ims[0], ax=ax[:,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\log_{10}$ Quantity",rotation=270,labelpad=20.0)
#plt.tight_layout()
#fig.suptitle("Spatial Cell Dynamics")
plt.savefig("Images/cell_dynamics_short_time.png",bbox_inches='tight')

ims = []
fig, ax = plt.subplots(4, 4, figsize=(10, 10),squeeze=True)
# Not DRY code, but quick.
for i, t in enumerate(times1):
    im = ax[0,i].imshow(np.log10(T_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[0,i].set_yticklabels([])

    ax[0,i].set_xticklabels([])

    ax[0,i].set_title(f"t={t_eval[t]:.1f} (days)")
    ims.append(im)

for i, t in enumerate(times1):
    im = ax[1,i].imshow(np.log10(I_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[1,i].set_yticklabels([])

    ims.append(im)

for i, t in enumerate(times1):
    im = ax[2,i].imshow(np.log10(I_star_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[2,i].set_yticklabels([])

    ims.append(im)

for i, t in enumerate(times1):
    im = ax[3,i].imshow(np.log10(R_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    if i!=0:
        ax[3,i].set_yticklabels([])

    ims.append(im)

fig.text(0.03, 0.8, r"Uninfected ($T$)", ha="center", va="center", fontsize=10)#, rotation=90)
fig.text(0.03, 0.6, r"Producing ($I$)", ha="center", va="center", fontsize=10)
fig.text(0.03, 0.4, r"Antiviral ($I^{*}$)", ha="center", va="center", fontsize=10)
fig.text(0.03, 0.2, r"Refractory ($R$)", ha="center", va="center", fontsize=10)

cbar = fig.colorbar(ims[0], ax=ax[:,:], orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\log_{10}$ Quantity",rotation=270,labelpad=20.0)
#plt.tight_layout()
#fig.suptitle("Spatial Cell Dynamics")
plt.savefig("Images/cell_dynamics_long_time.png",bbox_inches='tight')


## Refractory Cells
fig, ax = plt.subplots(1, 4, figsize=(16, 4))

epsilon = 1e-8
V_datasets = [np.log10(R_sol[i]+epsilon) for i in times2]
V_min = min([V.min() for V in V_datasets])
V_max = max([V.max() for V in V_datasets])

for i, t in enumerate(times2):
    im = ax[i].imshow(np.log10(R_sol[t]+epsilon), extent=[0, Lx, 0, Ly], origin="lower", cmap="viridis",vmin=V_min,vmax=V_max)
    ax[i].set_title(f"t={t_eval[t]:.1f} (days)")
    ax[i].set_xlabel("x")
    
ax[0].set_ylabel("y")
cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label(r"$\log_{10}$ Quantity",rotation=270,labelpad=20.0)
fig.suptitle("Refractory Cells")
plt.savefig("Images/refractory_cells.png")


## Aggregated Dynamics
print(V_sol.shape)
print(np.sum(V_sol,axis=(1,2)).shape)
plt.figure()
plt.plot(t_eval,np.clip(np.sum(V_sol,axis=(1,2)),a_min=1.0,a_max=None),label="Virions")
plt.plot(t_eval,np.clip(np.sum(F_sol,axis=(1,2)),a_min=1.0,a_max=None),label="Interferons")
plt.plot(t_eval,np.clip(np.sum(I_sol+I_star_sol,axis=(1,2)),a_min=1.0,a_max=None),label="Infected Cells")
plt.title("Aggregated Infection Dynamics")
plt.xlabel("time (days)")
plt.ylabel(r"Quantity")
plt.yscale("log")
plt.legend()
plt.savefig("Images/aggregated_virions_and_interferons.png")
